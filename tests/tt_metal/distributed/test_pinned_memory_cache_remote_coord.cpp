// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ---------------------------------------------------------------------------
// Minimal, reproduction of a multi-host crash in
// PinnedMemoryCache::compute_device_ids()  (tt_metal/distributed/pinned_memory_cache.cpp).
//
// THE BUG
//   compute_device_ids() loops over the requested mesh coordinates and, for any
//   coordinate that passes `view.contains(coord)`, calls `view.impl().get_device(coord)`:
//
//       for (const auto& coord : coordinate_range_set.coords()) {
//           if (view.contains(coord)) {                          // <-- only guard
//               if (auto* device = view.impl().get_device(coord)) {  // crashes on a remote coord
//                   device_ids.insert(device->id());
//               }
//           }
//       }
//
//   In a multi-host mesh every rank holds a view over the global mesh, so
//   `contains(coord)` is true for every coordinate on every rank, including the
//   coordinates owned by other hosts. For those, `get_device()` hits:
//       TT_FATAL(maybe_device.is_local(), "Cannot get device for remote device ...")
//   which throws; it is uncaught in compute_device_ids()/try_pin(), so the worker aborts.
//
//   `contains()` answers "is this coord in the global grid?" (true everywhere).
//   The extra needed guard is "is this coord local to this process?" (`is_local`).
//
//  Hardware needed to run: none
//   The only thing that matters is the state of the view: a coordinate that is
//   in-bounds but REMOTE. We build exactly that on one process by constructing a
//   MeshDeviceView with `MaybeRemote<IDevice*>::remote()` entries, no second
//   machine, no MPI, no real devices.
//
// HOW TO USE THIS TEST
//   Flip APPLY_IS_LOCAL_FIX below:
//     0  -> current main (BUG)    -> test FAILS  (TT_FATAL "Cannot get device for remote device")
//     1  -> is_local fix/workaround -> test PASSES (remote coords skipped gracefully)
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <set>
#include <vector>

#include <tt-metalium/device.hpp>
#include <tt-metalium/maybe_remote.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device_view.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <distributed/mesh_device_view_impl.hpp>

namespace tt::tt_metal::distributed {
namespace {

// ===========================================================================
//   >>> FLIP THIS to switch between the BROKEN and FIXED behavior <<<
//        0 = current main (BUG)    -> test FAILS
//        1 = proposed is_local fix -> test PASSES
#define APPLY_IS_LOCAL_FIX 0 
// ===========================================================================

// Faithful copy of the loop body of PinnedMemoryCache::compute_device_ids().
// (The real method is private and takes a MeshDevice&, so we cannot feed it a
// synthetic view directly; this is the identical logic, and the crashing call
// `view.impl().get_device(coord)` is the exact product call site.)
std::set<int> compute_device_ids(const MeshDeviceView& view, const MeshCoordinateRangeSet& ranges) {
    std::set<int> device_ids;
    for (const auto& coord : ranges.coords()) {
#if APPLY_IS_LOCAL_FIX
        // FIX/WORKAROUND: also require the coordinate to be local to this process.
        // is_local() is safe once contains() is true (it only throws out-of-bounds).
        if (view.contains(coord) && view.is_local(coord)) {
#else
        // CURRENT MAIN (BUG): no locality check -> get_device() crashes on a remote coord.
        if (view.contains(coord)) {
#endif
            if (auto* device = view.impl().get_device(coord)) {
                device_ids.insert(device->id());
            }
        }
    }
    return device_ids;
}

// Build a 1x2 MeshDeviceView whose coordinates are marked REMOTE -- i.e. the
// view a worker has onto devices owned by another host. No real IDevice* is
// needed: the constructor only dereferences LOCAL entries, so a remote view
// builds cleanly with no hardware.
MeshDeviceView make_remote_view() {
    const MeshShape shape(1, 2);
    const tt::tt_fabric::MeshId mesh_id{0};

    std::vector<MaybeRemote<IDevice*>> devices = {
        MaybeRemote<IDevice*>::remote(),  // coord (0, 0) -> owned by another host
        MaybeRemote<IDevice*>::remote(),  // coord (0, 1) -> owned by another host
    };
    std::vector<tt::tt_fabric::FabricNodeId> fabric_node_ids = {
        tt::tt_fabric::FabricNodeId(mesh_id, 0),
        tt::tt_fabric::FabricNodeId(mesh_id, 1),
    };
    return MeshDeviceView(shape, devices, fabric_node_ids);
}

// HEADLINE TEST -- behavior is controlled by APPLY_IS_LOCAL_FIX above.
//   APPLY_IS_LOCAL_FIX 0 -> FAILS: compute_device_ids() throws on the remote coord.
//   APPLY_IS_LOCAL_FIX 1 -> PASSES: remote coords are skipped, no crash.
TEST(PinnedMemoryCacheRemoteCoord, ComputeDeviceIdsMustNotCrashOnRemoteCoordinate) {
    MeshDeviceView view = make_remote_view();

    // Request the full 1x2 range -- exactly like a worker being asked to pin for
    // the whole mesh, including coordinates it does not own.
    MeshCoordinateRangeSet ranges{MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 1))};

    std::set<int> ids;
    EXPECT_NO_THROW(ids = compute_device_ids(view, ranges));

    // This view is fully remote, so the fixed version collects no local devices.
    // (In a partially-local mesh it would return exactly the local device ids.)
    EXPECT_TRUE(ids.empty());
}

// Supporting fact (always passes, independent of the toggle): an in-bounds
// remote coordinate passes contains() but is NOT local -- the exact gap that
// lets compute_device_ids() reach get_device() with a remote coordinate.
TEST(PinnedMemoryCacheRemoteCoord, ContainsIsTrueButCoordIsNotLocal) {
    MeshDeviceView view = make_remote_view();
    const MeshCoordinate remote_coord(0, 1);

    EXPECT_TRUE(view.contains(remote_coord));           // the only guard on main -> passes
    EXPECT_FALSE(view.is_local(remote_coord));   // the discriminator the fix adds
}

}  // namespace
}  // namespace tt::tt_metal::distributed
