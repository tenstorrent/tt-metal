# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import gc

import ttnn
from models.common.modules.lazy_weight import LazyWeight


def cleanup_ttnn_value(value):
    if value is None:
        return

    if isinstance(value, ttnn.Tensor):
        ttnn.deallocate(value)
        return

    if isinstance(value, dict):
        for nested_value in value.values():
            cleanup_ttnn_value(nested_value)
        return

    if isinstance(value, (list, tuple, set)):
        for nested_value in value:
            cleanup_ttnn_value(nested_value)


def cleanup_object_graph(obj, seen=None):
    if obj is None:
        return
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return
    seen.add(obj_id)

    if isinstance(obj, ttnn.Tensor):
        cleanup_ttnn_value(obj)
        return

    if isinstance(obj, LazyWeight):
        if obj._value is not None:
            cleanup_ttnn_value(obj._value)
            obj._value = None
        return

    if isinstance(obj, dict):
        for value in obj.values():
            cleanup_object_graph(value, seen)
        return

    if isinstance(obj, (list, tuple, set)):
        for value in obj:
            cleanup_object_graph(value, seen)
        return

    state = getattr(obj, "__dict__", None)
    if state is None:
        return

    for name, value in list(state.items()):
        cleanup_object_graph(value, seen)
        if isinstance(value, ttnn.Tensor):
            setattr(obj, name, None)

    if hasattr(obj, "_device_weights_loaded"):
        obj._device_weights_loaded = False


def cleanup_model_case(model, mesh_device):
    ttnn.synchronize_device(mesh_device)
    if model is not None:
        cleanup_object_graph(model)
    ttnn.synchronize_device(mesh_device)
    gc.collect()


def cleanup_dp_model_case(group, lanes, models, parent_mesh, submeshes):
    """Release one carved DP case and restore command-queue ownership to its parent."""

    failures = []

    def run(action):
        try:
            action()
        except Exception as error:
            failures.append(error)

    if group is not None:
        run(group.cleanup)
    else:
        for lane in lanes:
            run(lane.cleanup)

    for model, submesh in models:
        run(lambda model=model, submesh=submesh: cleanup_model_case(model, submesh))

    # A carved child owns a distinct MeshCommandQueue over its parent's physical
    # devices. Drain every live child through the parent before closing the child
    # handles and returning command-queue ownership to the fixture-owned parent.
    run(parent_mesh.quiesce_devices)
    for submesh in submeshes:
        run(lambda submesh=submesh: ttnn.close_mesh_device(submesh))

    if failures:
        primary = failures[0]
        add_note = getattr(primary, "add_note", None)
        if add_note is not None:
            for failure in failures[1:]:
                add_note(f"additional DP cleanup failure: {type(failure).__name__}: {failure}")
        raise primary
