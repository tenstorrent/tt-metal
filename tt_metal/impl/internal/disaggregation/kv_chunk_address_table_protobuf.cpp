// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/internal/disaggregation/kv_chunk_address_table_protobuf.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <map>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <unistd.h>

#include <google/protobuf/text_format.h>

#include "protobuf/kv_chunk_address_table.pb.h"

namespace tt::tt_metal::internal::disaggregation {

namespace detail {

namespace {

// Run detection: largest chunk_step considered when looking for a periodic address-delta
// sequence within a (slot, layer) row. Covers block-cyclic layouts up to 64 banks.
constexpr uint32_t kMaxRunStep = 64;

// Dual-write threshold: while the estimated unrolled `entries` payload stays below this,
// STRIDED_ROWS configs also mirror every chunk into `entries` so pre-runs readers keep
// working (they ignore `runs` and the compression tag). Above it, runs-only — such a table
// exceeds protobuf's ~2GB message cap unrolled, so no entries-only reader could have
// consumed it anyway. Override for tests/canary.
constexpr uint64_t kDefaultDualWriteMaxBytes = 1ull << 30;

uint64_t dual_write_max_bytes() {
    if (const char* env = std::getenv("KV_CHUNK_TABLE_DUAL_WRITE_MAX_BYTES")) {
        try {
            return std::stoull(env);
        } catch (const std::exception&) {
            // fall through to the default
        }
    }
    return kDefaultDualWriteMaxBytes;
}

bool is_unset(const KvCacheLocation& loc) {
    return loc.noc_addr == 0 && loc.size_bytes == 0 && *loc.device_group_index == 0;
}

// The smallest s with the row's address-delta sequence periodic (addresses affine per residue
// class mod s), or 0 if none. s is capped at (n-1)/2 so the period is PROVEN by at least one
// repetition — any sequence trivially "fits" a period covering it once, which would compress
// nothing. 0 = no proven period -> the config cannot be STRIDED_ROWS.
uint32_t delta_period(std::span<const KvCacheLocation> row) {
    const uint32_t n = static_cast<uint32_t>(row.size());
    if (n <= 1) {
        return 1;
    }
    const uint32_t limit = std::min((n - 1) / 2, kMaxRunStep);
    for (uint32_t s = 1; s <= limit; s++) {
        bool ok = true;
        for (uint32_t i = s; i + 1 < n && ok; i++) {
            if (row[i + 1].noc_addr - row[i].noc_addr != row[i - s + 1].noc_addr - row[i - s].noc_addr) {
                ok = false;
            }
        }
        if (ok) {
            return s;
        }
    }
    return 0;
}

// Newest format_version this reader knows. 0 = legacy (pre-tag) files; 1 = compression tags.
// Bump when the wire format changes incompatibly; old readers must keep working via dual-write.
constexpr uint32_t kMaxKnownFormatVersion = 1;

// Wire conversion, with fail-closed validation of the declared tag.
ChunkCompression from_wire(::tt::disaggregation::proto::ChunkCompression c) {
    switch (c) {
        case ::tt::disaggregation::proto::UNROLLED: return ChunkCompression::kUnrolled;
        case ::tt::disaggregation::proto::STRIDED_ROWS: return ChunkCompression::kStridedRows;
        default:
            throw std::runtime_error(
                "KvChunkAddressTable proto declares unknown compression=" +
                std::to_string(static_cast<int>(c)) + " — written by a newer format; upgrade the reader");
    }
}

::tt::disaggregation::proto::ChunkCompression to_wire(ChunkCompression c) {
    return c == ChunkCompression::kStridedRows ? ::tt::disaggregation::proto::STRIDED_ROWS
                                               : ::tt::disaggregation::proto::UNROLLED;
}

void emit_entry(
    ::tt::disaggregation::proto::KvChunkAddressTable& pb,
    uint32_t c,
    uint32_t slot,
    uint32_t layer,
    uint32_t chunk,
    uint32_t chunk_n_tokens,
    const KvCacheLocation& loc) {
    auto* entry = pb.add_entries();
    entry->set_slot(slot);
    entry->set_layer(layer);
    entry->set_position(chunk * chunk_n_tokens);
    entry->set_noc_addr(loc.noc_addr);
    entry->set_size_bytes(loc.size_bytes);
    entry->set_device_group_index(*loc.device_group_index);
    entry->set_config_idx(c);
}

void emit_run(
    ::tt::disaggregation::proto::KvChunkAddressTable& pb,
    uint32_t c,
    uint32_t slot,
    uint32_t layer,
    uint32_t start_chunk,
    uint32_t step,
    uint32_t count,
    uint64_t base,
    int64_t stride,
    uint32_t size_bytes,
    uint32_t device_group_index) {
    auto* run = pb.add_runs();
    run->set_config_idx(c);
    run->set_slot(slot);
    run->set_layer(layer);
    run->set_start_chunk(start_chunk);
    run->set_chunk_step(step);
    run->set_count(count);
    run->set_base_noc_addr(base);
    run->set_addr_stride(stride);
    run->set_size_bytes(size_bytes);
    run->set_device_group_index(device_group_index);
}

// Detection pass for the UNROLLED export path: a config converts to STRIDED_ROWS iff EVERY
// populated row is dense (no unset holes) with uniform size/group and a proven delta period
// (see delta_period). All-unset rows are tolerated (they carry no data and read back zeroed),
// but at least one must exist — an all-unset config would emit zero runs, which import
// rejects as malformed, so it stays UNROLLED.
bool config_compressible(
    const KvChunkAddressTable::UnrolledGrid& map, uint32_t num_slots, uint32_t num_layers, uint32_t npc) {
    if (npc == 0) {
        return false;
    }
    bool any_populated = false;
    for (uint32_t slot = 0; slot < num_slots; slot++) {
        for (uint32_t layer = 0; layer < num_layers; layer++) {
            const auto row = map.lookup_range(layer, 0, npc, slot);
            if (std::all_of(row.begin(), row.end(), is_unset)) {
                continue;  // never-populated row: no runs, reads back zeroed
            }
            any_populated = true;
            if (std::any_of(row.begin(), row.end(), is_unset)) {
                return false;  // holes — can't cover densely
            }
            const auto& first = row.front();
            const bool uniform = std::all_of(row.begin(), row.end(), [&](const KvCacheLocation& l) {
                return l.size_bytes == first.size_bytes && l.device_group_index == first.device_group_index;
            });
            if (!uniform || delta_period(row) == 0) {
                return false;
            }
        }
    }
    return any_populated;
}

using Row = KvChunkAddressTable::StridedRowMap::Row;

// Emit one run per residue class of one populated row (count geometry is shared by both
// provenance paths: detected rows from an UnrolledGrid, stored rows from a StridedRowMap).
void emit_row_runs(
    ::tt::disaggregation::proto::KvChunkAddressTable& pb,
    uint32_t c,
    uint32_t slot,
    uint32_t layer,
    uint32_t npc,
    const Row& row) {
    for (uint32_t r = 0; r < row.step; r++) {
        const uint32_t count = (npc - r + row.step - 1) / row.step;
        emit_run(
            pb, c, slot, layer, r, row.step, count, row.bases[r], row.strides[r], row.size_bytes,
            *row.device_group_index);
    }
}

// Detect a populated unrolled row's strided structure as a Row. Call only when
// config_compressible() returned true — then delta_period() is guaranteed nonzero.
Row detect_row(std::span<const KvCacheLocation> row) {
    Row out;
    out.step = delta_period(row);
    out.size_bytes = row.front().size_bytes;
    out.device_group_index = row.front().device_group_index;
    out.bases.resize(out.step);
    out.strides.resize(out.step);
    for (uint32_t r = 0; r < out.step; r++) {
        const uint32_t count = (static_cast<uint32_t>(row.size()) - r + out.step - 1) / out.step;
        out.bases[r] = row[r].noc_addr;
        out.strides[r] = static_cast<int64_t>(count > 1 ? row[r + out.step].noc_addr - row[r].noc_addr : 0);
    }
    return out;
}

// Emit one run per residue class for every populated row of a config. Call only when
// config_compressible() returned true — rows are then hole-free, uniform, and periodic.
void emit_config_runs(
    ::tt::disaggregation::proto::KvChunkAddressTable& pb,
    const KvChunkAddressTable::UnrolledGrid& map,
    uint32_t c,
    uint32_t num_slots,
    uint32_t num_layers,
    uint32_t npc) {
    for (uint32_t slot = 0; slot < num_slots; slot++) {
        for (uint32_t layer = 0; layer < num_layers; layer++) {
            const auto row = map.lookup_range(layer, 0, npc, slot);
            if (std::all_of(row.begin(), row.end(), is_unset)) {
                continue;  // never-populated row: no runs, reads back zeroed
            }
            emit_row_runs(pb, c, slot, layer, npc, detect_row(row));
        }
    }
}

// Per-config payload emission, one overload per map representation. visit_map() instantiates
// the call per concrete type, so the dispatch is static (no runtime branch, no if-constexpr).

// StridedRowMap: already compressed in memory — serialize the stored rows, no detection.
// dual_write mirrors entries expanded back out of the map (same cost as the unrolled path):
// without it, a small dual-written table re-exported through a NEW reader would come back
// runs-only, silently dropping the mirror for any OLD reader downstream of the re-export.
void emit_config_payload(
    ::tt::disaggregation::proto::KvChunkAddressTable& pb,
    ::tt::disaggregation::proto::KvChunkConfig& pb_cfg,
    const KvChunkAddressTable::StridedRowMap& map,
    uint32_t c,
    const KvChunkAddressTableConfig& cfg,
    uint32_t npc,
    bool dual_write) {
    pb_cfg.set_compression(to_wire(ChunkCompression::kStridedRows));
    for (uint32_t slot = 0; slot < cfg.num_slots; slot++) {
        for (uint32_t layer = 0; layer < cfg.num_layers; layer++) {
            const auto& row = map.rows[static_cast<size_t>(slot) * cfg.num_layers + layer];
            if (row.step == 0) {
                continue;  // never-populated row
            }
            emit_row_runs(pb, c, slot, layer, npc, row);
        }
    }
    if (dual_write) {
        for (uint32_t slot = 0; slot < cfg.num_slots; slot++) {
            for (uint32_t layer = 0; layer < cfg.num_layers; layer++) {
                for (uint32_t chunk = 0; chunk < npc; chunk++) {
                    const auto loc = map.lookup(layer, chunk, slot);
                    if (!is_unset(loc)) {
                        emit_entry(pb, c, slot, layer, chunk, cfg.chunk_n_tokens, loc);
                    }
                }
            }
        }
    }
}

// UnrolledGrid: detect the stride structure; the config converts to STRIDED_ROWS only if every
// populated row compresses (per-config tag granularity), otherwise it stays UNROLLED. Entries
// are the sole payload for UNROLLED configs, or the old-reader mirror under dual-write.
void emit_config_payload(
    ::tt::disaggregation::proto::KvChunkAddressTable& pb,
    ::tt::disaggregation::proto::KvChunkConfig& pb_cfg,
    const KvChunkAddressTable::UnrolledGrid& map,
    uint32_t c,
    const KvChunkAddressTableConfig& cfg,
    uint32_t npc,
    bool dual_write) {
    const bool convertible = config_compressible(map, cfg.num_slots, cfg.num_layers, npc);
    if (convertible) {
        pb_cfg.set_compression(to_wire(ChunkCompression::kStridedRows));
        emit_config_runs(pb, map, c, cfg.num_slots, cfg.num_layers, npc);
    } else {
        pb_cfg.set_compression(to_wire(ChunkCompression::kUnrolled));
    }
    if (!convertible || dual_write) {
        for (uint32_t slot = 0; slot < cfg.num_slots; slot++) {
            for (uint32_t layer = 0; layer < cfg.num_layers; layer++) {
                const auto row = map.lookup_range(layer, 0, npc, slot);
                for (uint32_t chunk = 0; chunk < npc; chunk++) {
                    if (!is_unset(row[chunk])) {
                        emit_entry(pb, c, slot, layer, chunk, cfg.chunk_n_tokens, row[chunk]);
                    }
                }
            }
        }
    }
}

}  // namespace

::tt::disaggregation::proto::KvChunkAddressTable to_proto_message(const KvChunkAddressTable& table) {
    ::tt::disaggregation::proto::KvChunkAddressTable pb;
    pb.set_format_version(kMaxKnownFormatVersion);
    char hostname[256];
    if (::gethostname(hostname, sizeof(hostname)) == 0) {
        hostname[sizeof(hostname) - 1] = '\0';
        pb.set_origin_host(hostname);
    }

    // Dual-write decision needs the total unrolled size up front (grid-equivalent count).
    const uint64_t total_chunks = table.total_entries();
    const bool dual_write = total_chunks * 32 <= dual_write_max_bytes();  // ~32 B per entry on the wire

    for (uint32_t c = 0; c < table.num_configs(); c++) {
        const auto& cfg = table.config(c);
        const uint32_t npc = table.num_position_chunks(c);
        auto* pb_cfg = pb.add_configs();
        pb_cfg->set_name(table.config_name(c));
        pb_cfg->set_num_layers(cfg.num_layers);
        pb_cfg->set_max_sequence_length(cfg.max_sequence_length);
        pb_cfg->set_num_slots(cfg.num_slots);
        pb_cfg->set_chunk_n_tokens(cfg.chunk_n_tokens);
        pb_cfg->set_chunk_size_bytes(cfg.chunk_size_bytes);

        // Static dispatch per map representation (overload set above — one instantiation per
        // alternative; no runtime branch).
        table.visit_map(c, [&](const auto& map) { emit_config_payload(pb, *pb_cfg, map, c, cfg, npc, dual_write); });
    }

    const auto& cfg0 = table.config(0);
    pb.set_num_layers(cfg0.num_layers);
    pb.set_max_sequence_length(cfg0.max_sequence_length);
    pb.set_num_slots(cfg0.num_slots);
    pb.set_chunk_n_tokens(cfg0.chunk_n_tokens);
    pb.set_chunk_size_bytes(cfg0.chunk_size_bytes);

    for (size_t i = 0; i < table.num_device_groups(); i++) {
        const auto& group = table.get_device_group(DeviceGroupIndex{static_cast<uint32_t>(i)});
        auto* pb_group = pb.add_device_groups();
        for (const auto& fnid : group.fabric_node_ids) {
            auto* pb_fnid = pb_group->add_fabric_node_ids();
            pb_fnid->set_mesh_id(*fnid.mesh_id);
            pb_fnid->set_chip_id(fnid.chip_id);
        }
    }

    // Export host mappings, deduplicating across device groups.
    // Only hosts for nodes that appear in at least one device group are exported.
    std::unordered_set<tt::tt_fabric::FabricNodeId> exported_hosts;
    for (size_t i = 0; i < table.num_device_groups(); i++) {
        const auto& group = table.get_device_group(DeviceGroupIndex{static_cast<uint32_t>(i)});
        for (const auto& fnid : group.fabric_node_ids) {
            if (table.has_host(fnid) && exported_hosts.insert(fnid).second) {
                auto* pb_host = pb.add_fabric_node_hosts();
                pb_host->set_mesh_id(*fnid.mesh_id);
                pb_host->set_chip_id(fnid.chip_id);
                pb_host->set_host_name(table.get_host(fnid));
            }
        }
    }

    return pb;
}

KvChunkAddressTable from_proto_message(const ::tt::disaggregation::proto::KvChunkAddressTable& pb) {
    // Fail closed on newer formats (old readers ignore this field and read the dual-written
    // entries — the intended transition path; a runs-only file yields an empty-but-valid
    // legacy table there, which is why rollout upgrades readers first).
    if (pb.format_version() > kMaxKnownFormatVersion) {
        throw std::runtime_error(
            "KvChunkAddressTable format_version=" + std::to_string(pb.format_version()) +
            " is newer than this reader supports (max " + std::to_string(kMaxKnownFormatVersion) +
            ") — upgrade the reader");
    }
    // Reconstruct configs. `configs` (field 9) is authoritative when present;
    // otherwise fall back to the legacy single-config scalar fields. Entries are
    // placed by config NAME (idx_to_name) so they land correctly even if the map
    // constructor reassigns ids by sorted-key order.
    std::map<std::string, KvChunkAddressTableConfig> configs;
    std::vector<std::string> idx_to_name;
    std::vector<ChunkCompression> idx_to_compression;
    if (pb.configs_size() > 0) {
        idx_to_name.reserve(pb.configs_size());
        for (const auto& pb_cfg : pb.configs()) {
            KvChunkAddressTableConfig cfg{
                .num_layers = pb_cfg.num_layers(),
                .max_sequence_length = pb_cfg.max_sequence_length(),
                .num_slots = pb_cfg.num_slots(),
                .chunk_n_tokens = pb_cfg.chunk_n_tokens(),
                .chunk_size_bytes = pb_cfg.chunk_size_bytes(),
            };
            if (!configs.emplace(pb_cfg.name(), cfg).second) {
                throw std::runtime_error("duplicate config name '" + pb_cfg.name() + "' in KvChunkAddressTable proto");
            }
            idx_to_name.push_back(pb_cfg.name());
            idx_to_compression.push_back(from_wire(pb_cfg.compression()));  // throws on unknown tag
        }
    } else {
        configs.emplace(
            "0",
            KvChunkAddressTableConfig{
                .num_layers = pb.num_layers(),
                .max_sequence_length = pb.max_sequence_length(),
                .num_slots = pb.num_slots(),
                .chunk_n_tokens = pb.chunk_n_tokens(),
                .chunk_size_bytes = pb.chunk_size_bytes(),
            });
        idx_to_name.push_back("0");
        idx_to_compression.push_back(ChunkCompression::kUnrolled);  // legacy files: entries only
    }
    KvChunkAddressTable table(configs);

    for (const auto& pb_group : pb.device_groups()) {
        std::vector<tt::tt_fabric::FabricNodeId> fnids;
        fnids.reserve(pb_group.fabric_node_ids_size());
        for (const auto& pb_fnid : pb_group.fabric_node_ids()) {
            fnids.emplace_back(tt::tt_fabric::MeshId{pb_fnid.mesh_id()}, pb_fnid.chip_id());
        }
        table.add_device_group(std::move(fnids));
    }

    for (const auto& pb_host : pb.fabric_node_hosts()) {
        tt::tt_fabric::FabricNodeId fnid(tt::tt_fabric::MeshId{pb_host.mesh_id()}, pb_host.chip_id());
        table.set_fabric_node_host(fnid, pb_host.host_name());
    }

    // Entries apply to UNROLLED configs only; for STRIDED_ROWS configs they are the dual-write
    // mirror and runs are authoritative.
    for (const auto& entry : pb.entries()) {
        if (entry.config_idx() >= idx_to_name.size()) {
            throw std::runtime_error("entry config_idx out of range in KvChunkAddressTable proto");
        }
        if (idx_to_compression[entry.config_idx()] != ChunkCompression::kUnrolled) {
            continue;
        }
        KvCacheLocation loc{
            .noc_addr = entry.noc_addr(),
            .size_bytes = entry.size_bytes(),
            .device_group_index = DeviceGroupIndex{entry.device_group_index()},
        };
        table.set(entry.layer(), entry.position(), entry.slot(), loc, idx_to_name[entry.config_idx()]);
    }

    // Strided runs: instantiate a StridedRowMap per STRIDED_ROWS config. Rows must tile
    // densely — each populated row has one run per residue class 0..step-1 with the exact
    // counts implied by the geometry; anything else is rejected as malformed.
    {
        struct RowAccum {
            uint32_t step = 0;
            uint32_t size_bytes = 0;
            uint32_t group = 0;
            std::vector<uint64_t> bases;  // indexed by start_chunk (residue)
            std::vector<int64_t> strides;
            std::vector<bool> seen;
        };
        std::map<std::tuple<uint32_t, uint32_t, uint32_t>, RowAccum> rows;
        for (const auto& run : pb.runs()) {
            if (run.config_idx() >= idx_to_name.size()) {
                throw std::runtime_error("run config_idx out of range in KvChunkAddressTable proto");
            }
            const uint32_t cid = table.config_id_of(idx_to_name[run.config_idx()]);
            if (idx_to_compression[run.config_idx()] != ChunkCompression::kStridedRows) {
                throw std::runtime_error("run targets an UNROLLED config in KvChunkAddressTable proto");
            }
            if (run.chunk_step() == 0 || run.chunk_step() > table.num_position_chunks(cid)) {
                throw std::runtime_error("run chunk_step out of range in KvChunkAddressTable proto");
            }
            if (run.count() == 0) {
                throw std::runtime_error("run count must be >= 1 in KvChunkAddressTable proto");
            }
            const auto& cfg = table.config(cid);
            if (run.layer() >= cfg.num_layers || run.slot() >= cfg.num_slots) {
                throw std::runtime_error("run layer/slot out of range in KvChunkAddressTable proto");
            }
            if (run.start_chunk() >= run.chunk_step()) {
                throw std::runtime_error("run start_chunk must be < chunk_step in KvChunkAddressTable proto");
            }
            const uint32_t npc = table.num_position_chunks(cid);
            const uint32_t expect = (npc - run.start_chunk() + run.chunk_step() - 1) / run.chunk_step();
            if (run.count() != expect) {
                throw std::runtime_error("run count does not tile the row in KvChunkAddressTable proto");
            }
            auto& acc = rows[{run.config_idx(), run.slot(), run.layer()}];
            if (acc.bases.empty()) {
                acc.step = run.chunk_step();
                acc.size_bytes = run.size_bytes();
                acc.group = run.device_group_index();
                acc.bases.resize(run.chunk_step());
                acc.strides.resize(run.chunk_step());
                acc.seen.resize(run.chunk_step());
            } else if (
                acc.step != run.chunk_step() || acc.size_bytes != run.size_bytes() ||
                acc.group != run.device_group_index()) {
                throw std::runtime_error("inconsistent runs for one row in KvChunkAddressTable proto");
            }
            if (acc.seen[run.start_chunk()]) {
                throw std::runtime_error("duplicate run residue in KvChunkAddressTable proto");
            }
            acc.seen[run.start_chunk()] = true;
            acc.bases[run.start_chunk()] = run.base_noc_addr();
            acc.strides[run.start_chunk()] = run.addr_stride();
        }
        for (const auto& [key, acc] : rows) {
            if (std::any_of(acc.seen.begin(), acc.seen.end(), [](bool s) { return !s; })) {
                throw std::runtime_error("row runs do not cover all residues in KvChunkAddressTable proto");
            }
        }
        std::map<uint32_t, KvChunkAddressTable::StridedRowMap> strided_maps;
        for (const auto& [key, acc] : rows) {
            const auto [cidx, slot, layer] = key;
            const uint32_t cid = table.config_id_of(idx_to_name[cidx]);
            auto& map = strided_maps[cid];
            if (map.rows.empty()) {
                const auto& cfg = table.config(cid);
                map.num_slots = cfg.num_slots;
                map.num_layers = cfg.num_layers;
                map.num_position_chunks = table.num_position_chunks(cid);
                map.rows.resize(static_cast<size_t>(cfg.num_slots) * cfg.num_layers);
            }
            auto& row = map.rows[static_cast<size_t>(slot) * table.config(cid).num_layers + layer];
            row.step = acc.step;
            row.size_bytes = acc.size_bytes;
            row.device_group_index = DeviceGroupIndex{acc.group};
            row.bases = acc.bases;
            row.strides = acc.strides;
        }
        for (auto& [cid, map] : strided_maps) {
            table.install_strided_map(cid, std::move(map));
        }
        // A config tagged STRIDED_ROWS with no runs at all is an empty table declaration —
        // almost certainly a corrupt/truncated payload; fail loudly.
        for (uint32_t i = 0; i < idx_to_compression.size(); i++) {
            if (idx_to_compression[i] == ChunkCompression::kStridedRows && !strided_maps.count(i)) {
                // …unless every row is legitimately unset: our exporter only tags configs whose
                // rows all compress, and an all-unset config exports UNROLLED, so this is corrupt.
                throw std::runtime_error(
                    "config '" + idx_to_name[i] + "' tagged STRIDED_ROWS but has no runs — malformed proto");
            }
        }
    }

    return table;
}

}  // namespace detail

// --- Binary wire format ---

std::string export_to_protobuf(const KvChunkAddressTable& table) {
    auto pb = detail::to_proto_message(table);
    std::string out;
    if (!pb.SerializeToString(&out)) {
        throw std::runtime_error("Failed to serialize KvChunkAddressTable to protobuf");
    }
    return out;
}

void export_to_protobuf_file(const KvChunkAddressTable& table, const std::string& path) {
    std::string data = export_to_protobuf(table);
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

KvChunkAddressTable import_from_protobuf(const std::string& data) {
    ::tt::disaggregation::proto::KvChunkAddressTable pb;
    if (!pb.ParseFromString(data)) {
        throw std::runtime_error("Failed to parse protobuf data as KvChunkAddressTable");
    }
    return detail::from_proto_message(pb);
}

KvChunkAddressTable import_from_protobuf_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }
    std::string data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return import_from_protobuf(data);
}

// --- Text format (debug only) ---

std::string export_to_protobuf_text(const KvChunkAddressTable& table) {
    auto pb = detail::to_proto_message(table);
    std::string out;
    if (!google::protobuf::TextFormat::PrintToString(pb, &out)) {
        throw std::runtime_error("Failed to serialize KvChunkAddressTable to protobuf text format");
    }
    return out;
}

void export_to_protobuf_text_file(const KvChunkAddressTable& table, const std::string& path) {
    std::string text = export_to_protobuf_text(table);
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    out << text;
}

KvChunkAddressTable import_from_protobuf_text(const std::string& text) {
    ::tt::disaggregation::proto::KvChunkAddressTable pb;
    if (!google::protobuf::TextFormat::ParseFromString(text, &pb)) {
        throw std::runtime_error("Failed to parse protobuf text format as KvChunkAddressTable");
    }
    return detail::from_proto_message(pb);
}

KvChunkAddressTable import_from_protobuf_text_file(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + path);
    }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return import_from_protobuf_text(text);
}

}  // namespace tt::tt_metal::internal::disaggregation
