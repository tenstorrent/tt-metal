// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <iterator>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include <tt_stl/span.hpp>
#include <tt-metalium/device_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_device_options.hpp>

namespace tt::tt_metal::distributed {

// A set of independently usable 1x1 MeshDevices, one per physical chip, opened together. Each member is a unit mesh:
// it has its own command queues, allocator and program cache, and can be driven from its own thread.
//
// The group owns its members. A member (a `MeshDevice&` obtained from at() or by iteration) must not be used after the
// group is closed or destroyed, and everything created on a member (buffers, programs, sockets, ...) must be released
// before then. Opening the chips together (instead of one by one) is what lets them share one initialization pass of
// fabric and dispatch firmware, which a chip opened alone cannot do.
//
// Replaces MeshDevice::create_unit_meshes, which returned a std::map of shared_ptr<MeshDevice> and thereby let any
// member outlive its siblings.
//
// Only the group is movable; the members never move.
class UnitMeshGroup {
public:
    // Opens a unit mesh on each chip in `chip_ids` against the default MetalContext. `chip_ids` must be non-empty. Use
    // MetalEnv::create_unit_meshes to open against a specific MetalEnv.
    static UnitMeshGroup open(ttsl::Span<const ChipId> chip_ids, const MeshDeviceOptions& options = {});

    // An empty group, for use as a placeholder to be move-assigned to.
    UnitMeshGroup() = default;
    UnitMeshGroup(const UnitMeshGroup&) = delete;
    UnitMeshGroup& operator=(const UnitMeshGroup&) = delete;
    UnitMeshGroup(UnitMeshGroup&& other) noexcept;
    UnitMeshGroup& operator=(UnitMeshGroup&& other) noexcept;
    // Closes the members that are still open.
    ~UnitMeshGroup();

    // Closes every member and the devices behind them, then leaves the group empty. Idempotent.
    void close();

    // The number of members. Zero after close() or after being moved from.
    size_t size() const { return members_.size(); }
    bool empty() const { return members_.empty(); }

    bool contains(ChipId chip_id) const { return members_.contains(chip_id); }

    // The unit mesh on `chip_id`. Throws std::runtime_error when `chip_id` is not a member.
    MeshDevice& at(ChipId chip_id) const;

    // The member chips in ascending order.
    std::vector<ChipId> chip_ids() const;

    // Iterates the members in ascending chip order; dereferencing yields `std::pair<ChipId, MeshDevice&>`, so that
    // `for (auto [chip_id, mesh] : group)` works.
    class Iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = std::pair<ChipId, MeshDevice&>;
        using difference_type = std::ptrdiff_t;
        using pointer = void;
        using reference = value_type;

        reference operator*() const { return {it_->first, *it_->second}; }
        Iterator& operator++() {
            ++it_;
            return *this;
        }
        Iterator operator++(int) {
            Iterator previous = *this;
            ++it_;
            return previous;
        }
        bool operator==(const Iterator& other) const { return it_ == other.it_; }
        bool operator!=(const Iterator& other) const { return it_ != other.it_; }

    private:
        friend class UnitMeshGroup;
        using Underlying = std::map<ChipId, std::shared_ptr<MeshDevice>>::const_iterator;
        explicit Iterator(Underlying it) : it_(it) {}
        Underlying it_;
    };

    Iterator begin() const { return Iterator(members_.begin()); }
    Iterator end() const { return Iterator(members_.end()); }

private:
    friend class MeshDeviceImpl;

    // Members are shared_ptr for now because MeshDevice is still created and handed out as shared_ptr. This is a detail
    // of this class and not part of its interface.
    using Members = std::map<ChipId, std::shared_ptr<MeshDevice>>;
    explicit UnitMeshGroup(Members members) : members_(std::move(members)) {}

    Members members_;
};

}  // namespace tt::tt_metal::distributed
