// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/profiler/perf_counter_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <set>
#include <utility>

#include <common/TracyTTDeviceData.hpp>
#include <tt-logger/tt-logger.hpp>

#include "tools/profiler/perf_counters.hpp"

namespace tt::tt_metal::profiler_perf_counters {

namespace {

// (value, ref_cnt) for one counter type on one core.
using CounterVal = std::pair<double, double>;
// counter_type enum value -> (value, ref_cnt). One entry per counter on a (op, core).
using CoreCounters = std::map<uint16_t, CounterVal>;
using CoreKey = std::pair<int32_t, int32_t>;  // (core_x, core_y)
using OpPivot = std::map<CoreKey, CoreCounters>;
using Pivot = std::map<experimental::ProgramExecutionUID, OpPivot>;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

bool is_finite(double v) { return std::isfinite(v); }

// pandas groupby().min/median/max/mean — all skip NaN; empty (all-NaN) op yields "no value".
struct Stat {
    bool any = false;
    double mn = kNaN, med = kNaN, mx = kNaN, avg = kNaN;
};

Stat reduce_finite(std::vector<double>& vals) {
    Stat s;
    vals.erase(std::remove_if(vals.begin(), vals.end(), [](double v) { return !is_finite(v); }), vals.end());
    if (vals.empty()) {
        return s;
    }
    s.any = true;
    std::sort(vals.begin(), vals.end());
    s.mn = vals.front();
    s.mx = vals.back();
    double sum = 0.0;
    for (double v : vals) {
        sum += v;
    }
    s.avg = sum / static_cast<double>(vals.size());
    const size_t n = vals.size();
    s.med = (n % 2 == 1) ? vals[n / 2] : (vals[n / 2 - 1] + vals[n / 2]) / 2.0;
    return s;
}

std::optional<CounterVal> get(const CoreCounters& cc, uint16_t type) {
    auto it = cc.find(type);
    if (it == cc.end()) {
        return std::nullopt;
    }
    return it->second;
}

// Sum of a counter's value over every (op, core) — mirrors df["counter"].sum(), used by the
// Python fallback gates (e.g. "packer unused" -> use the dest-read grant rate instead).
double global_value_sum(const Pivot& pivot, uint16_t type) {
    double sum = 0.0;
    for (const auto& [uid, op] : pivot) {
        for (const auto& [core, cc] : op) {
            if (auto v = get(cc, type)) {
                sum += v->first;
            }
        }
    }
    return sum;
}

bool any_core_has(const Pivot& pivot, uint16_t type) {
    for (const auto& [uid, op] : pivot) {
        for (const auto& [core, cc] : op) {
            if (cc.count(type)) {
                return true;
            }
        }
    }
    return false;
}

// A per-core metric: returns the metric value for one core, or nullopt to drop it (NaN).
using CoreFn = std::function<std::optional<double>(const CoreCounters&)>;

// value / ref_cnt * scale (ref<=0 -> drop, matching x/0 -> inf -> NaN).
CoreFn util_fn(uint16_t type, double scale = 100.0) {
    return [type, scale](const CoreCounters& cc) -> std::optional<double> {
        auto v = get(cc, type);
        if (!v || v->second <= 0.0) {
            return std::nullopt;
        }
        double r = v->first / v->second * scale;
        return is_finite(r) ? std::optional<double>(r) : std::nullopt;
    };
}

// numerator.value / denominator.value * scale (both required; den<=0 -> drop).
CoreFn ratio_fn(uint16_t num, uint16_t den, double scale = 100.0) {
    return [num, den, scale](const CoreCounters& cc) -> std::optional<double> {
        auto n = get(cc, num);
        auto d = get(cc, den);
        if (!n || !d || d->first == 0.0) {
            return std::nullopt;
        }
        double r = n->first / d->first * scale;
        return is_finite(r) ? std::optional<double>(r) : std::nullopt;
    };
}

// (total - counter) / total * 100, clipped at 0 (both required; total<=0 -> drop).
CoreFn complement_fn(uint16_t counter, uint16_t total) {
    return [counter, total](const CoreCounters& cc) -> std::optional<double> {
        auto c = get(cc, counter);
        auto t = get(cc, total);
        if (!c || !t || t->first == 0.0) {
            return std::nullopt;
        }
        double r = (t->first - c->first) / t->first * 100.0;
        if (!is_finite(r)) {
            return std::nullopt;
        }
        return std::max(0.0, r);
    };
}

// (a.value + b.value) / 2 / a.ref_cnt * scale — two-channel util (a.ref as denominator).
CoreFn avg_channel_fn(uint16_t a, uint16_t b, double scale = 100.0) {
    return [a, b, scale](const CoreCounters& cc) -> std::optional<double> {
        auto va = get(cc, a);
        auto vb = get(cc, b);
        if (!va || !vb || va->second <= 0.0) {
            return std::nullopt;
        }
        double r = (va->first + vb->first) / 2.0 / va->second * scale;
        return is_finite(r) ? std::optional<double>(r) : std::nullopt;
    };
}

// ((r0-g0)+(r1-g1)) / (r0+r1) * 100, clipped at 0 — two-channel back-pressure.
CoreFn backpressure_fn(uint16_t r0, uint16_t r1, uint16_t g0, uint16_t g1) {
    return [r0, r1, g0, g1](const CoreCounters& cc) -> std::optional<double> {
        auto vr0 = get(cc, r0);
        auto vr1 = get(cc, r1);
        auto vg0 = get(cc, g0);
        auto vg1 = get(cc, g1);
        if (!vr0 || !vr1 || !vg0 || !vg1) {
            return std::nullopt;
        }
        double denom = vr0->first + vr1->first;
        if (denom == 0.0) {
            return std::nullopt;
        }
        double r = ((vr0->first - vg0->first) + (vr1->first - vg1->first)) / denom * 100.0;
        if (!is_finite(r)) {
            return std::nullopt;
        }
        return std::max(0.0, r);
    };
}

// Accumulates per-op column values and tracks which columns ever got data (active set).
class ColumnSink {
public:
    explicit ColumnSink(const Pivot& pivot) : pivot_(pivot) {}

    // Emit the 4 stat columns (Min/Median/Max/Avg) for a base metric. suffix is " (%)" or "".
    void emit_stat(const std::string& base, const std::string& suffix, const CoreFn& fn) {
        for (const auto& [uid, op] : pivot_) {
            std::vector<double> vals;
            vals.reserve(op.size());
            for (const auto& [core, cc] : op) {
                if (auto v = fn(cc)) {
                    vals.push_back(*v);
                }
            }
            Stat s = reduce_finite(vals);
            if (!s.any) {
                continue;
            }
            set(uid, base + " Min" + suffix, s.mn);
            set(uid, base + " Median" + suffix, s.med);
            set(uid, base + " Max" + suffix, s.mx);
            set(uid, base + " Avg" + suffix, s.avg);
        }
    }

    // Emit stats from a precomputed per-core value map (for metrics not expressible as a single CoreFn,
    // e.g. the skipna mean of two channel efficiencies).
    void emit_stat_from(
        const std::string& base,
        const std::string& suffix,
        const std::map<experimental::ProgramExecutionUID, std::vector<double>>& per_op_vals) {
        for (const auto& [uid, raw] : per_op_vals) {
            std::vector<double> vals = raw;
            Stat s = reduce_finite(vals);
            if (!s.any) {
                continue;
            }
            set(uid, base + " Min" + suffix, s.mn);
            set(uid, base + " Median" + suffix, s.med);
            set(uid, base + " Max" + suffix, s.mx);
            set(uid, base + " Avg" + suffix, s.avg);
        }
    }

    void set(const experimental::ProgramExecutionUID& uid, const std::string& col, double val) {
        if (!is_finite(val)) {
            return;
        }
        // Mirror python float formatting closely enough for downstream float() parsing.
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.6g", val);
        values_[uid][col] = buf;
        active_.insert(col);
    }

    bool active(const std::string& col) const { return active_.count(col) > 0; }

    PerfCounterColumns finalize(const std::vector<std::string>& canonical_order) {
        PerfCounterColumns out;
        // Emit the FULL canonical schema (not just the columns active on this device) whenever any
        // counter value exists. writeProgramsPerfResultsToCSV appends one block per device to a shared
        // CSV with a single header row; a per-device-variable column set would misalign the appended
        // blocks. A fixed schema keeps every block aligned — empty cells for an uncaptured group are
        // trimmed downstream by the Python active-header pass.
        if (!values_.empty()) {
            out.active_headers = canonical_order;
        }
        out.values_per_uid = std::move(values_);
        return out;
    }

private:
    const Pivot& pivot_;
    std::map<experimental::ProgramExecutionUID, std::map<std::string, std::string>> values_;
    std::set<std::string> active_;
};

// Canonical column order — MUST match tools/tracy/perf_counter_analysis.py PERF_COUNTER_CSV_HEADERS
// so the emitted cpp_device_perf_report.csv columns line up with the legacy layout.
const std::vector<std::string>& canonical_headers() {
    // Build mechanically from base names + (Min/Median/Max/Avg) + suffix so the order and spelling
    // exactly match tools/tracy/perf_counter_analysis.py PERF_COUNTER_CSV_HEADERS. The first three
    // compute metrics interleave a grid-normalized column, so they are listed explicitly.
    static const std::vector<std::string> headers = [] {
        std::vector<std::string> h;
        auto quad = [&](const std::string& base, const std::string& suffix) {
            h.push_back(base + " Min" + suffix);
            h.push_back(base + " Median" + suffix);
            h.push_back(base + " Max" + suffix);
            h.push_back(base + " Avg" + suffix);
        };
        // Compute utilizations: Min/Median/Max then the grid-normalized average (legacy column order).
        h.push_back("SFPU Util Min (%)");
        h.push_back("SFPU Util Median (%)");
        h.push_back("SFPU Util Max (%)");
        h.push_back("Avg SFPU util on full grid (%)");
        h.push_back("FPU Util Min (%)");
        h.push_back("FPU Util Median (%)");
        h.push_back("FPU Util Max (%)");
        h.push_back("Avg FPU util on full grid (%)");
        h.push_back("MATH Util Min (%)");
        h.push_back("MATH Util Median (%)");
        h.push_back("MATH Util Max (%)");
        h.push_back("Avg Math util on full grid (%)");
        const char* pct[] = {
            "Unpacker0 Write Efficiency",
            "Unpacker1 Write Efficiency",
            "Unpacker Write Efficiency",
            "Packer Efficiency",
            "FPU Execution Efficiency",
            "Math Pipeline Utilization",
            "Math-to-Pack Handoff Efficiency",
            "Unpacker-to-Math Data Flow",
            "Thread 0 Stall Rate",
            "Thread 1 Stall Rate",
            "Thread 2 Stall Rate",
            "SrcA Valid Wait",
            "SrcB Valid Wait",
            "SrcA Clear Wait",
            "SrcB Clear Wait",
            "Math Idle Wait T1",
            "Pack Idle Wait T2",
            "Unpack Idle Wait T0",
            "Semaphore Zero Wait T0",
            "Semaphore Zero Wait T1",
            "Semaphore Zero Wait T2",
            "Semaphore Full Wait T0",
            "Semaphore Full Wait T1",
            "Semaphore Full Wait T2",
            "Data Hazard Stall Rate",
            "Fidelity Stall Rate",
            "HiFi Fraction"};
        for (const char* b : pct) {
            quad(b, " (%)");
        }
        quad("Avg HF Cycles Per Instrn", "");
        const char* pct2[] = {
            "L1 Unpacker Port Util",
            "L1 TDMA Bundle Util",
            "NOC Ring 0 Outgoing Util",
            "NOC Ring 0 Incoming Util",
            "NOC Ring 1 Outgoing Util",
            "NOC Ring 1 Incoming Util",
            "L1 Packer Port Util",
            "NOC Ring 0 Outgoing Backpressure",
            "NOC Ring 0 Incoming Backpressure",
            "NOC Ring 1 Outgoing Backpressure",
            "NOC Ring 1 Incoming Backpressure",
            "L1 Unpacker Backpressure",
            "L1 Packer Port Backpressure",
            "SrcA Write Port Blocked Rate",
            "SrcA Write Overwrite Blocked Rate",
            "SrcB Write Overwrite Blocked Rate",
            "Dest Read Backpressure",
            "Math Dest Write Port Stall Rate",
            "Math Scoreboard Stall Rate"};
        for (const char* b : pct2) {
            quad(b, " (%)");
        }
        quad("T0 Instrn Issue Rate", "");
        quad("T1 Instrn Issue Rate", "");
        quad("T2 Instrn Issue Rate", "");
        const char* pct3[] = {
            "CFG Instrn Avail Rate T0",
            "SYNC Instrn Avail Rate T0",
            "THCON Instrn Avail Rate T0",
            "MOVE Instrn Avail Rate T0",
            "MATH Instrn Avail Rate T1",
            "UNPACK Instrn Avail Rate T0",
            "PACK Instrn Avail Rate T2",
            "SrcB Write Port Blocked Rate",
            "SrcA Write Actual Efficiency",
            "SrcB Write Actual Efficiency",
            "Packer Engine 0 Util",
            "Packer Engine 1 Util",
            "Packer Engine 2 Util",
            "MMIO Idle Wait T0",
            "SFPU Idle Wait T1",
            "THCON Idle Wait T0",
            "MOVE Idle Wait T0",
            "RISC Core L1 Util",
            "L1 Total Bandwidth Util",
            "L1 Read vs Write Ratio",
            "NOC Ring 0 Asymmetry",
            "L1 Contention Index",
            "Unpacker L1 Efficiency",
            "Packer L1 Efficiency",
            "NOC vs Compute Balance",
            "TDMA vs NOC L1 Share",
            "Stall Overlap T0",
            "Stall Overlap T1",
            "Stall Overlap T2",
            "Packer Load Imbalance",
            "Compute-to-Unpack Ratio"};
        for (const char* b : pct3) {
            quad(b, " (%)");
        }
        return h;
    }();
    return headers;
}

}  // namespace

PerfCounterColumns computePerfCounterColumns(
    const std::vector<std::reference_wrapper<const tracy::TTDeviceMarker>>& device_markers,
    uint32_t total_compute_cores,
    const std::map<experimental::ProgramExecutionUID, double>& kernel_cycles_by_uid) {
    // 1) Pivot the perf-counter markers into per-(op, core, counter) (value, ref_cnt).
    Pivot pivot;
    size_t dbg_counter_markers = 0, dbg_type0 = 0;
    for (const auto& marker_ref : device_markers) {
        const tracy::TTDeviceMarker& marker = marker_ref.get();
        if (marker.marker_id != PERF_COUNTER_PROFILER_ID) {
            continue;
        }
        ++dbg_counter_markers;
        const PerfCounter pc(marker.data, marker.data_high);
        const uint16_t type = static_cast<uint16_t>(pc.counter_type);
        if (type == 0) {  // UNDEF
            ++dbg_type0;
            continue;
        }
        const experimental::ProgramExecutionUID uid{marker.runtime_host_id, marker.trace_id, marker.trace_id_counter};
        const CoreKey core{static_cast<int32_t>(marker.core_x), static_cast<int32_t>(marker.core_y)};
        // First marker wins per (op, core, counter) — matches pivot_table(aggfunc="first").
        auto& slot = pivot[uid][core];
        slot.emplace(type, CounterVal{static_cast<double>(pc.counter_value), static_cast<double>(pc.ref_cnt)});
    }

    // Only chatter on a counter-capture run (markers present); silent on the default no-counter path.
    if (dbg_counter_markers > 0) {
        log_info(
            tt::LogMetal,
            "perf-counter post-process: {} counter markers over {} ops ({} undefined-type skipped)",
            dbg_counter_markers,
            pivot.size(),
            dbg_type0);
    }

    ColumnSink sink(pivot);
    if (pivot.empty()) {
        return sink.finalize(canonical_headers());
    }

    // Global fallback gates (computed across the whole run, like df["x"].sum()).
    const double packer_busy_sum = global_value_sum(pivot, PACKER_BUSY);
    const double math_started_sum = global_value_sum(pivot, MATH_INSTRN_STARTED);

    // 2) Core compute utilizations + grid-normalized averages.
    sink.emit_stat("SFPU Util", " (%)", util_fn(SFPU_COUNTER));
    sink.emit_stat("FPU Util", " (%)", util_fn(FPU_COUNTER));
    sink.emit_stat("MATH Util", " (%)", util_fn(MATH_COUNTER));

    // "Avg X util on full grid (%)" = (Σ_cores counter.value / total_compute_cores) / kernel_cycles * 100.
    auto emit_grid_avg = [&](uint16_t type, const std::string& col) {
        if (total_compute_cores == 0) {
            return;
        }
        for (const auto& [uid, op] : pivot) {
            auto cyc_it = kernel_cycles_by_uid.find(uid);
            if (cyc_it == kernel_cycles_by_uid.end() || cyc_it->second <= 0.0) {
                continue;
            }
            double core_sum = 0.0;
            bool any = false;
            for (const auto& [core, cc] : op) {
                if (auto v = get(cc, type)) {
                    core_sum += v->first;
                    any = true;
                }
            }
            if (!any) {
                continue;
            }
            double avg_count = core_sum / static_cast<double>(total_compute_cores);
            sink.set(uid, col, avg_count / cyc_it->second * 100.0);
        }
    };
    emit_grid_avg(SFPU_COUNTER, "Avg SFPU util on full grid (%)");
    emit_grid_avg(FPU_COUNTER, "Avg FPU util on full grid (%)");
    emit_grid_avg(MATH_COUNTER, "Avg Math util on full grid (%)");

    // 3) Unpacker / packer / math-pipeline efficiencies (with the python fallback logic).
    sink.emit_stat("Unpacker0 Write Efficiency", " (%)", ratio_fn(SRCA_WRITE_ACTUAL, UNPACK0_BUSY_THREAD0));
    sink.emit_stat("Unpacker1 Write Efficiency", " (%)", ratio_fn(SRCB_WRITE_NOT_BLOCKED_PORT, UNPACK1_BUSY_THREAD0));

    // Unpacker Write Efficiency = per-core skipna mean of unpack0/unpack1 efficiencies.
    {
        CoreFn u0 = ratio_fn(SRCA_WRITE_ACTUAL, UNPACK0_BUSY_THREAD0);
        CoreFn u1 = ratio_fn(SRCB_WRITE_NOT_BLOCKED_PORT, UNPACK1_BUSY_THREAD0);
        std::map<experimental::ProgramExecutionUID, std::vector<double>> per_op;
        for (const auto& [uid, op] : pivot) {
            for (const auto& [core, cc] : op) {
                auto a = u0(cc);
                auto b = u1(cc);
                if (a && b) {
                    per_op[uid].push_back((*a + *b) / 2.0);
                } else if (a) {
                    per_op[uid].push_back(*a);
                } else if (b) {
                    per_op[uid].push_back(*b);
                }
            }
        }
        sink.emit_stat_from("Unpacker Write Efficiency", " (%)", per_op);
    }

    // Packer Efficiency: packer_dest_read / packer_busy, else dest_read_granted_0 / packer_dest_read.
    if (packer_busy_sum > 0.0) {
        sink.emit_stat("Packer Efficiency", " (%)", ratio_fn(PACKER_DEST_READ_AVAILABLE, PACKER_BUSY));
    } else if (any_core_has(pivot, DEST_READ_GRANTED_0)) {
        sink.emit_stat("Packer Efficiency", " (%)", ratio_fn(DEST_READ_GRANTED_0, PACKER_DEST_READ_AVAILABLE));
    }

    sink.emit_stat("FPU Execution Efficiency", " (%)", ratio_fn(FPU_COUNTER, FPU_INSTRN_AVAILABLE_1));

    // Math Pipeline Utilization only when math was issued during the run.
    if (math_started_sum > 0.0) {
        sink.emit_stat("Math Pipeline Utilization", " (%)", ratio_fn(MATH_INSTRN_STARTED, MATH_INSTRN_AVAILABLE));
    }

    // Math-to-Pack Handoff: available_math / packer_busy, else available_math / its own ref_cnt.
    if (packer_busy_sum > 0.0) {
        sink.emit_stat("Math-to-Pack Handoff Efficiency", " (%)", ratio_fn(AVAILABLE_MATH, PACKER_BUSY));
    } else if (any_core_has(pivot, AVAILABLE_MATH)) {
        sink.emit_stat("Math-to-Pack Handoff Efficiency", " (%)", util_fn(AVAILABLE_MATH));
    }

    // Unpacker-to-Math Data Flow = ((srca_avail + srcb_avail)/2) / ((unpack0 + unpack1)/2) * 100.
    {
        std::map<experimental::ProgramExecutionUID, std::vector<double>> per_op;
        for (const auto& [uid, op] : pivot) {
            for (const auto& [core, cc] : op) {
                auto sa = get(cc, SRCA_WRITE_AVAILABLE);
                auto sb = get(cc, SRCB_WRITE_AVAILABLE);
                auto u0 = get(cc, UNPACK0_BUSY_THREAD0);
                auto u1 = get(cc, UNPACK1_BUSY_THREAD0);
                if (!sa || !sb || !u0 || !u1) {
                    continue;
                }
                double denom = (u0->first + u1->first) / 2.0;
                if (denom == 0.0) {
                    continue;
                }
                double r = (sa->first + sb->first) / 2.0 / denom * 100.0;
                if (is_finite(r)) {
                    per_op[uid].push_back(r);
                }
            }
        }
        sink.emit_stat_from("Unpacker-to-Math Data Flow", " (%)", per_op);
    }

    // ---- Remaining metrics (faithful port of compute_perf_counter_metrics) ----
    // value of a counter on a core (no ref), for the composite formulas.
    auto val = [](const CoreCounters& cc, uint16_t t) -> std::optional<double> {
        auto it = cc.find(t);
        if (it == cc.end()) {
            return std::nullopt;
        }
        return it->second.first;
    };

    // Thread stall rates + pipeline/idle/semaphore waits — all value/ref_cnt*100.
    sink.emit_stat("Thread 0 Stall Rate", " (%)", util_fn(THREAD_STALLS_0));
    sink.emit_stat("Thread 1 Stall Rate", " (%)", util_fn(THREAD_STALLS_1));
    sink.emit_stat("Thread 2 Stall Rate", " (%)", util_fn(THREAD_STALLS_2));
    sink.emit_stat("SrcA Valid Wait", " (%)", util_fn(WAITING_FOR_SRCA_VALID));
    sink.emit_stat("SrcB Valid Wait", " (%)", util_fn(WAITING_FOR_SRCB_VALID));
    sink.emit_stat("SrcA Clear Wait", " (%)", util_fn(WAITING_FOR_SRCA_CLEAR));
    sink.emit_stat("SrcB Clear Wait", " (%)", util_fn(WAITING_FOR_SRCB_CLEAR));
    sink.emit_stat("Math Idle Wait T1", " (%)", util_fn(WAITING_FOR_MATH_IDLE_1));
    sink.emit_stat("Pack Idle Wait T2", " (%)", util_fn(WAITING_FOR_PACK_IDLE_2));
    sink.emit_stat("Unpack Idle Wait T0", " (%)", util_fn(WAITING_FOR_UNPACK_IDLE_0));
    sink.emit_stat("Semaphore Zero Wait T0", " (%)", util_fn(WAITING_FOR_NONZERO_SEM_0));
    sink.emit_stat("Semaphore Zero Wait T1", " (%)", util_fn(WAITING_FOR_NONZERO_SEM_1));
    sink.emit_stat("Semaphore Zero Wait T2", " (%)", util_fn(WAITING_FOR_NONZERO_SEM_2));
    sink.emit_stat("Semaphore Full Wait T0", " (%)", util_fn(WAITING_FOR_NONFULL_SEM_0));
    sink.emit_stat("Semaphore Full Wait T1", " (%)", util_fn(WAITING_FOR_NONFULL_SEM_1));
    sink.emit_stat("Semaphore Full Wait T2", " (%)", util_fn(WAITING_FOR_NONFULL_SEM_2));

    sink.emit_stat("Data Hazard Stall Rate", " (%)", complement_fn(DATA_HAZARD_STALLS_MOVD2A, MATH_INSTRN_AVAILABLE));
    sink.emit_stat("Fidelity Stall Rate", " (%)", ratio_fn(MATH_FIDELITY_STALL, MATH_INSTRN_AVAILABLE));

    // HiFi Fraction + Avg HF cycles/instrn (needs all three HF cycle-grant counters).
    if (any_core_has(pivot, MATH_INSTRN_HF_1_CYCLE) && any_core_has(pivot, MATH_INSTRN_HF_2_CYCLE) &&
        any_core_has(pivot, MATH_INSTRN_HF_4_CYCLE)) {
        sink.emit_stat("HiFi Fraction", " (%)", [&](const CoreCounters& cc) -> std::optional<double> {
            double h1 = val(cc, MATH_INSTRN_HF_1_CYCLE).value_or(0.0);
            double h2 = val(cc, MATH_INSTRN_HF_2_CYCLE).value_or(0.0);
            double h4 = val(cc, MATH_INSTRN_HF_4_CYCLE).value_or(0.0);
            double total = h1 + h2 + h4;
            if (total <= 0.0) {
                return std::nullopt;
            }
            return (h2 + h4) / total * 100.0;
        });
        sink.emit_stat("Avg HF Cycles Per Instrn", "", [&](const CoreCounters& cc) -> std::optional<double> {
            double h1 = val(cc, MATH_INSTRN_HF_1_CYCLE).value_or(0.0);
            double h2 = val(cc, MATH_INSTRN_HF_2_CYCLE).value_or(0.0);
            double h4 = val(cc, MATH_INSTRN_HF_4_CYCLE).value_or(0.0);
            double total = h1 + h2 + h4;
            if (total <= 0.0) {
                return std::nullopt;
            }
            return (h1 + 2.0 * h2 + 4.0 * h4) / total;
        });
    }

    // L1 Bank 0 utilizations.
    sink.emit_stat("L1 Unpacker Port Util", " (%)", util_fn(L1_0_UNPACKER_0));
    sink.emit_stat("L1 TDMA Bundle Util", " (%)", avg_channel_fn(L1_0_TDMA_BUNDLE_0_RISC, L1_0_TDMA_BUNDLE_1_TRISC));
    sink.emit_stat(
        "NOC Ring 0 Outgoing Util", " (%)", avg_channel_fn(L1_0_NOC_RING0_OUTGOING_0, L1_0_NOC_RING0_OUTGOING_1));
    sink.emit_stat(
        "NOC Ring 0 Incoming Util", " (%)", avg_channel_fn(L1_0_NOC_RING0_INCOMING_0, L1_0_NOC_RING0_INCOMING_1));
    // L1 Bank 1.
    sink.emit_stat(
        "NOC Ring 1 Outgoing Util", " (%)", avg_channel_fn(L1_1_NOC_RING1_OUTGOING_0, L1_1_NOC_RING1_OUTGOING_1));
    sink.emit_stat(
        "NOC Ring 1 Incoming Util", " (%)", avg_channel_fn(L1_1_NOC_RING1_INCOMING_0, L1_1_NOC_RING1_INCOMING_1));
    // L1 Packer port (arch-specific): unified packer (BH) else port-1 ECC/pack1 (WH).
    const uint16_t packer_port =
        any_core_has(pivot, L1_0_UNIFIED_PACKER) ? L1_0_UNIFIED_PACKER : L1_0_UNPACKER_1_ECC_PACK1;
    if (any_core_has(pivot, packer_port)) {
        sink.emit_stat("L1 Packer Port Util", " (%)", util_fn(packer_port));
    }

    // NOC back-pressure (two-channel).
    sink.emit_stat(
        "NOC Ring 0 Outgoing Backpressure",
        " (%)",
        backpressure_fn(
            L1_0_NOC_RING0_OUTGOING_0,
            L1_0_NOC_RING0_OUTGOING_1,
            L1_0_NOC_RING0_OUTGOING_0_GRANT,
            L1_0_NOC_RING0_OUTGOING_1_GRANT));
    sink.emit_stat(
        "NOC Ring 0 Incoming Backpressure",
        " (%)",
        backpressure_fn(
            L1_0_NOC_RING0_INCOMING_0,
            L1_0_NOC_RING0_INCOMING_1,
            L1_0_NOC_RING0_INCOMING_0_GRANT,
            L1_0_NOC_RING0_INCOMING_1_GRANT));
    sink.emit_stat(
        "NOC Ring 1 Outgoing Backpressure",
        " (%)",
        backpressure_fn(
            L1_1_NOC_RING1_OUTGOING_0,
            L1_1_NOC_RING1_OUTGOING_1,
            L1_1_NOC_RING1_OUTGOING_0_GRANT,
            L1_1_NOC_RING1_OUTGOING_1_GRANT));
    sink.emit_stat(
        "NOC Ring 1 Incoming Backpressure",
        " (%)",
        backpressure_fn(
            L1_1_NOC_RING1_INCOMING_0,
            L1_1_NOC_RING1_INCOMING_1,
            L1_1_NOC_RING1_INCOMING_0_GRANT,
            L1_1_NOC_RING1_INCOMING_1_GRANT));

    // L1 Unpacker Backpressure — only when grant/req actually track each other (BH signal-mismatch guard:
    // median grant/req over cores with req>0 must exceed 0.1).
    if (any_core_has(pivot, L1_0_UNPACKER_0) && any_core_has(pivot, L1_0_UNPACKER_0_GRANT)) {
        std::vector<double> ratios;
        for (const auto& [uid, op] : pivot) {
            for (const auto& [core, cc] : op) {
                auto req = val(cc, L1_0_UNPACKER_0);
                auto grant = val(cc, L1_0_UNPACKER_0_GRANT);
                if (req && grant && *req > 0.0) {
                    ratios.push_back(*grant / *req);
                }
            }
        }
        double median_ratio = 0.0;
        if (!ratios.empty()) {
            std::sort(ratios.begin(), ratios.end());
            size_t n = ratios.size();
            median_ratio = (n % 2 == 1) ? ratios[n / 2] : (ratios[n / 2 - 1] + ratios[n / 2]) / 2.0;
        }
        if (median_ratio > 0.1) {
            sink.emit_stat("L1 Unpacker Backpressure", " (%)", [&](const CoreCounters& cc) -> std::optional<double> {
                auto req = val(cc, L1_0_UNPACKER_0);
                auto grant = val(cc, L1_0_UNPACKER_0_GRANT);
                if (!req || !grant || *req == 0.0) {
                    return std::nullopt;
                }
                double r = (*req - *grant) / *req * 100.0;
                return is_finite(r) ? std::optional<double>(std::max(0.0, r)) : std::nullopt;
            });
        }
    }

    // L1 Packer Port Backpressure — single (req-grant)/req, clipped; packer port is arch-specific.
    if (any_core_has(pivot, packer_port) && any_core_has(pivot, L1_0_PORT1_GRANT)) {
        sink.emit_stat(
            "L1 Packer Port Backpressure", " (%)", [&, packer_port](const CoreCounters& cc) -> std::optional<double> {
                auto req = val(cc, packer_port);
                auto grant = val(cc, L1_0_PORT1_GRANT);
                if (!req || !grant || *req == 0.0) {
                    return std::nullopt;
                }
                double r = (*req - *grant) / *req * 100.0;
                return is_finite(r) ? std::optional<double>(std::max(0.0, r)) : std::nullopt;
            });
    }

    // Write-port blocked rates (complement, clipped).
    sink.emit_stat("SrcA Write Port Blocked Rate", " (%)", complement_fn(SRCA_WRITE_ACTUAL, SRCA_WRITE_AVAILABLE));
    sink.emit_stat(
        "SrcA Write Overwrite Blocked Rate", " (%)", complement_fn(SRCA_WRITE_NOT_BLOCKED_OVR, SRCA_WRITE_AVAILABLE));
    sink.emit_stat("SrcB Write Overwrite Blocked Rate", " (%)", complement_fn(SRCB_WRITE_ACTUAL, SRCB_WRITE_AVAILABLE));

    // Dest Read Backpressure — single (req-grant)/req, NOT clipped (matches python).
    sink.emit_stat("Dest Read Backpressure", " (%)", [&](const CoreCounters& cc) -> std::optional<double> {
        auto req = val(cc, PACKER_DEST_READ_AVAILABLE);
        auto grant = val(cc, DEST_READ_GRANTED_0);
        if (!req || !grant || *req == 0.0) {
            return std::nullopt;
        }
        double r = (*req - *grant) / *req * 100.0;
        return is_finite(r) ? std::optional<double>(r) : std::nullopt;
    });

    // Math Dest Write Port Stall Rate — (avail - unstalled)/avail, no clip; skip if counter dead run-wide.
    if (global_value_sum(pivot, MATH_NOT_STALLED_DEST_WR_PORT) > 0.0) {
        sink.emit_stat("Math Dest Write Port Stall Rate", " (%)", [&](const CoreCounters& cc) -> std::optional<double> {
            auto avail = val(cc, MATH_INSTRN_AVAILABLE);
            auto unstalled = val(cc, MATH_NOT_STALLED_DEST_WR_PORT);
            if (!avail || !unstalled || *avail == 0.0) {
                return std::nullopt;
            }
            double r = (*avail - *unstalled) / *avail * 100.0;
            return is_finite(r) ? std::optional<double>(r) : std::nullopt;
        });
    }
    // Math Scoreboard Stall Rate — (avail - available_math)/avail, no clip.
    sink.emit_stat("Math Scoreboard Stall Rate", " (%)", [&](const CoreCounters& cc) -> std::optional<double> {
        auto avail = val(cc, MATH_INSTRN_AVAILABLE);
        auto unstalled = val(cc, AVAILABLE_MATH);
        if (!avail || !unstalled || *avail == 0.0) {
            return std::nullopt;
        }
        double r = (*avail - *unstalled) / *avail * 100.0;
        return is_finite(r) ? std::optional<double>(r) : std::nullopt;
    });

    // Per-thread instruction issue rates (per cycle, scale=1, no % suffix).
    sink.emit_stat("T0 Instrn Issue Rate", "", util_fn(THREAD_INSTRUCTIONS_0, 1.0));
    sink.emit_stat("T1 Instrn Issue Rate", "", util_fn(THREAD_INSTRUCTIONS_1, 1.0));
    sink.emit_stat("T2 Instrn Issue Rate", "", util_fn(THREAD_INSTRUCTIONS_2, 1.0));

    // Per-type instruction availability rates.
    sink.emit_stat("CFG Instrn Avail Rate T0", " (%)", util_fn(CFG_INSTRN_AVAILABLE_0));
    sink.emit_stat("SYNC Instrn Avail Rate T0", " (%)", util_fn(SYNC_INSTRN_AVAILABLE_0));
    sink.emit_stat("THCON Instrn Avail Rate T0", " (%)", util_fn(THCON_INSTRN_AVAILABLE_0));
    sink.emit_stat("MOVE Instrn Avail Rate T0", " (%)", util_fn(MOVE_INSTRN_AVAILABLE_0));
    sink.emit_stat("MATH Instrn Avail Rate T1", " (%)", util_fn(FPU_INSTRN_AVAILABLE_1));
    sink.emit_stat("UNPACK Instrn Avail Rate T0", " (%)", util_fn(UNPACK_INSTRN_AVAILABLE_0));
    sink.emit_stat("PACK Instrn Avail Rate T2", " (%)", util_fn(PACK_INSTRN_AVAILABLE_2));

    sink.emit_stat(
        "SrcB Write Port Blocked Rate", " (%)", complement_fn(SRCB_WRITE_NOT_BLOCKED_PORT, SRCB_WRITE_AVAILABLE));
    sink.emit_stat("SrcA Write Actual Efficiency", " (%)", ratio_fn(SRCA_WRITE_ACTUAL, SRCA_WRITE_AVAILABLE));
    sink.emit_stat("SrcB Write Actual Efficiency", " (%)", ratio_fn(SRCB_WRITE_NOT_BLOCKED_PORT, SRCB_WRITE_AVAILABLE));

    // Packer engine granularity (WH only).
    sink.emit_stat("Packer Engine 0 Util", " (%)", util_fn(PACKER_BUSY_0));
    sink.emit_stat("Packer Engine 1 Util", " (%)", util_fn(PACKER_BUSY_1));
    sink.emit_stat("Packer Engine 2 Util", " (%)", util_fn(PACKER_BUSY_2));

    // Low-priority idle waits + RISC-core L1.
    sink.emit_stat("MMIO Idle Wait T0", " (%)", util_fn(WAITING_FOR_MMIO_IDLE_0));
    sink.emit_stat("SFPU Idle Wait T1", " (%)", util_fn(WAITING_FOR_SFPU_IDLE_1));
    sink.emit_stat("THCON Idle Wait T0", " (%)", util_fn(WAITING_FOR_THCON_IDLE_0));
    sink.emit_stat("MOVE Idle Wait T0", " (%)", util_fn(WAITING_FOR_MOVE_IDLE_0));
    sink.emit_stat("RISC Core L1 Util", " (%)", util_fn(L1_1_RISC_CORE));

    // L1 composite metrics (gated on the bank-0 port set being present).
    if (any_core_has(pivot, L1_0_UNPACKER_0) && any_core_has(pivot, L1_0_NOC_RING0_OUTGOING_0)) {
        sink.emit_stat(
            "L1 Total Bandwidth Util", " (%)", [&, packer_port](const CoreCounters& cc) -> std::optional<double> {
                const uint16_t ports[] = {
                    L1_0_UNPACKER_0,
                    packer_port,
                    L1_0_TDMA_BUNDLE_0_RISC,
                    L1_0_TDMA_BUNDLE_1_TRISC,
                    L1_0_NOC_RING0_OUTGOING_0,
                    L1_0_NOC_RING0_OUTGOING_1,
                    L1_0_NOC_RING0_INCOMING_0,
                    L1_0_NOC_RING0_INCOMING_1};
                double total = 0.0;
                for (uint16_t p : ports) {
                    if (auto x = val(cc, p)) {
                        total += *x;
                    }
                }
                auto ref = get(cc, L1_0_UNPACKER_0);
                if (!ref || ref->second <= 0.0) {
                    return std::nullopt;
                }
                double r = total / (8.0 * ref->second) * 100.0;
                return is_finite(r) ? std::optional<double>(r) : std::nullopt;
            });
        sink.emit_stat(
            "L1 Read vs Write Ratio", " (%)", [&, packer_port](const CoreCounters& cc) -> std::optional<double> {
                auto u = val(cc, L1_0_UNPACKER_0);
                auto o0 = val(cc, L1_0_NOC_RING0_OUTGOING_0);
                auto o1 = val(cc, L1_0_NOC_RING0_OUTGOING_1);
                auto pk = val(cc, packer_port);
                auto i0 = val(cc, L1_0_NOC_RING0_INCOMING_0);
                auto i1 = val(cc, L1_0_NOC_RING0_INCOMING_1);
                double reads = u.value_or(0.0) + o0.value_or(0.0) + o1.value_or(0.0);
                double writes = pk.value_or(0.0) + i0.value_or(0.0) + i1.value_or(0.0);
                double total = reads + writes;
                if (total == 0.0) {
                    return std::nullopt;
                }
                return reads / total * 100.0;
            });
        sink.emit_stat("NOC Ring 0 Asymmetry", " (%)", [&](const CoreCounters& cc) -> std::optional<double> {
            double out =
                val(cc, L1_0_NOC_RING0_OUTGOING_0).value_or(0.0) + val(cc, L1_0_NOC_RING0_OUTGOING_1).value_or(0.0);
            double in =
                val(cc, L1_0_NOC_RING0_INCOMING_0).value_or(0.0) + val(cc, L1_0_NOC_RING0_INCOMING_1).value_or(0.0);
            double total = out + in;
            if (total == 0.0) {
                return std::nullopt;
            }
            return out / total * 100.0;
        });
        sink.emit_stat("TDMA vs NOC L1 Share", " (%)", [&](const CoreCounters& cc) -> std::optional<double> {
            double tdma =
                val(cc, L1_0_TDMA_BUNDLE_0_RISC).value_or(0.0) + val(cc, L1_0_TDMA_BUNDLE_1_TRISC).value_or(0.0);
            double noc =
                val(cc, L1_0_NOC_RING0_OUTGOING_0).value_or(0.0) + val(cc, L1_0_NOC_RING0_OUTGOING_1).value_or(0.0) +
                val(cc, L1_0_NOC_RING0_INCOMING_0).value_or(0.0) + val(cc, L1_0_NOC_RING0_INCOMING_1).value_or(0.0);
            double total = tdma + noc;
            if (total == 0.0) {
                return std::nullopt;
            }
            return tdma / total * 100.0;
        });
    }

    // L1 Contention Index — mean back-pressure across the active bank-0 (req,grant) pairs.
    if (any_core_has(pivot, L1_0_UNPACKER_0_GRANT) && any_core_has(pivot, L1_0_NOC_RING0_OUTGOING_0_GRANT)) {
        const std::pair<uint16_t, uint16_t> bp_pairs[] = {
            {L1_0_UNPACKER_0, L1_0_UNPACKER_0_GRANT},
            {L1_0_NOC_RING0_OUTGOING_0, L1_0_NOC_RING0_OUTGOING_0_GRANT},
            {L1_0_NOC_RING0_OUTGOING_1, L1_0_NOC_RING0_OUTGOING_1_GRANT},
            {L1_0_NOC_RING0_INCOMING_0, L1_0_NOC_RING0_INCOMING_0_GRANT},
            {L1_0_NOC_RING0_INCOMING_1, L1_0_NOC_RING0_INCOMING_1_GRANT}};
        std::vector<std::pair<uint16_t, uint16_t>> present;
        for (const auto& p : bp_pairs) {
            if (any_core_has(pivot, p.first) && any_core_has(pivot, p.second)) {
                present.push_back(p);
            }
        }
        if (!present.empty()) {
            sink.emit_stat(
                "L1 Contention Index", " (%)", [&, present](const CoreCounters& cc) -> std::optional<double> {
                    double sum = 0.0;
                    for (const auto& p : present) {
                        auto req = val(cc, p.first);
                        auto grant = val(cc, p.second);
                        if (!req || !grant || *req == 0.0) {
                            return std::nullopt;  // a present pair with no req on this core -> NaN (python align)
                        }
                        double bp = (*req - *grant) / *req * 100.0;
                        if (!is_finite(bp)) {
                            return std::nullopt;
                        }
                        sum += std::max(0.0, bp);
                    }
                    return sum / static_cast<double>(present.size());
                });
        }
    }

    sink.emit_stat("Unpacker L1 Efficiency", " (%)", ratio_fn(L1_0_UNPACKER_0_GRANT, UNPACK0_BUSY_THREAD0));
    sink.emit_stat("Packer L1 Efficiency", " (%)", ratio_fn(L1_0_PORT1_GRANT, PACKER_BUSY));

    // NOC vs Compute Balance — NOC / (FPU + NOC).
    if (any_core_has(pivot, FPU_COUNTER) && any_core_has(pivot, L1_0_NOC_RING0_OUTGOING_0)) {
        sink.emit_stat("NOC vs Compute Balance", " (%)", [&](const CoreCounters& cc) -> std::optional<double> {
            double noc =
                val(cc, L1_0_NOC_RING0_OUTGOING_0).value_or(0.0) + val(cc, L1_0_NOC_RING0_OUTGOING_1).value_or(0.0) +
                val(cc, L1_0_NOC_RING0_INCOMING_0).value_or(0.0) + val(cc, L1_0_NOC_RING0_INCOMING_1).value_or(0.0);
            auto fpu = val(cc, FPU_COUNTER);
            if (!fpu) {
                return std::nullopt;
            }
            double total = *fpu + noc;
            if (total == 0.0) {
                return std::nullopt;
            }
            return noc / total * 100.0;
        });
    }

    // Stall-cause overlap factor — Σ(reason waits) / total stalls (ratio, can exceed 1).
    for (int t = 0; t < 3; ++t) {
        const uint16_t stalls = static_cast<uint16_t>(THREAD_STALLS_0 + t);
        const uint16_t reasons[] = {
            static_cast<uint16_t>(WAITING_FOR_THCON_IDLE_0 + t),
            static_cast<uint16_t>(WAITING_FOR_UNPACK_IDLE_0 + t),
            static_cast<uint16_t>(WAITING_FOR_PACK_IDLE_0 + t),
            static_cast<uint16_t>(WAITING_FOR_MATH_IDLE_0 + t),
            static_cast<uint16_t>(WAITING_FOR_NONZERO_SEM_0 + t),
            static_cast<uint16_t>(WAITING_FOR_NONFULL_SEM_0 + t),
            static_cast<uint16_t>(WAITING_FOR_MOVE_IDLE_0 + t),
            static_cast<uint16_t>(WAITING_FOR_MMIO_IDLE_0 + t),
            static_cast<uint16_t>(WAITING_FOR_SFPU_IDLE_0 + t)};
        bool all_reasons = any_core_has(pivot, stalls);
        for (uint16_t r : reasons) {
            all_reasons = all_reasons && any_core_has(pivot, r);
        }
        if (!all_reasons) {
            continue;
        }
        std::vector<uint16_t> reason_vec(std::begin(reasons), std::end(reasons));
        sink.emit_stat(
            "Stall Overlap T" + std::to_string(t),
            " (%)",
            [&, stalls, reason_vec](const CoreCounters& cc) -> std::optional<double> {
                auto total = val(cc, stalls);
                if (!total || *total == 0.0) {
                    return std::nullopt;
                }
                double reason_sum = 0.0;
                for (uint16_t r : reason_vec) {
                    auto rv = val(cc, r);
                    if (!rv) {
                        return std::nullopt;
                    }
                    reason_sum += *rv;
                }
                double res = reason_sum / *total;
                return is_finite(res) ? std::optional<double>(res) : std::nullopt;
            });
    }

    // Packer Load Imbalance — (max-min)/max across the 3 engine busies + aggregate PACKER_BUSY.
    if (any_core_has(pivot, PACKER_BUSY_0) && any_core_has(pivot, PACKER_BUSY_1) &&
        any_core_has(pivot, PACKER_BUSY_2) && any_core_has(pivot, PACKER_BUSY)) {
        sink.emit_stat("Packer Load Imbalance", " (%)", [&](const CoreCounters& cc) -> std::optional<double> {
            const uint16_t busies[] = {PACKER_BUSY_0, PACKER_BUSY_1, PACKER_BUSY_2, PACKER_BUSY};
            double mx = -1.0, mn = 0.0;
            bool first = true;
            for (uint16_t b : busies) {
                auto v = val(cc, b);
                if (!v) {
                    return std::nullopt;
                }
                if (first) {
                    mx = mn = *v;
                    first = false;
                } else {
                    mx = std::max(mx, *v);
                    mn = std::min(mn, *v);
                }
            }
            if (mx <= 0.0) {
                return std::nullopt;
            }
            return (mx - mn) / mx * 100.0;
        });
    }

    // Compute-to-Unpack Ratio — MATH / (unpack0 + unpack1).
    sink.emit_stat("Compute-to-Unpack Ratio", " (%)", [&](const CoreCounters& cc) -> std::optional<double> {
        auto math = val(cc, MATH_COUNTER);
        auto u0 = val(cc, UNPACK0_BUSY_THREAD0);
        auto u1 = val(cc, UNPACK1_BUSY_THREAD0);
        if (!math || !u0 || !u1) {
            return std::nullopt;
        }
        double denom = *u0 + *u1;
        if (denom == 0.0) {
            return std::nullopt;
        }
        return *math / denom * 100.0;
    });

    return sink.finalize(canonical_headers());
}

}  // namespace tt::tt_metal::profiler_perf_counters
