# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Prepared, borrowed I/O for serialized traces on command queue zero."""

import ttnn


class PreparedTraceIO:
    """Own persistent buffers before capture and share only compatible lifetimes.

    A group identifies a model lane and a semantic role. Callers may share a
    group only when its consumers finish (or enqueue a copy) before the next
    writer. Inputs whose state must survive a variant switch need separate groups.
    Tensor specs include padding, tile shape and memory configuration; topology
    additionally distinguishes replicated tensors from mesh-sharded tensors.
    """

    def __init__(self):
        self._buffers = []
        self._frozen = False

    def require_preparation(self):
        if self._frozen:
            raise RuntimeError(
                "Trace I/O preparation is closed after the first capture. "
                "Warm all decode modes and prefill buckets before capturing; "
                "release live traces and reinitialize the model and generator to add an unprepared variant."
            )

    def freeze(self):
        self._frozen = True

    def _reserve(self, tensor, mesh, group, allocate):
        self.require_preparation()
        spec, topology = tensor.spec, tensor.tensor_topology()
        for saved_group, saved_mesh, saved_spec, saved_topology, buffer in self._buffers:
            if group == saved_group and mesh == saved_mesh and spec == saved_spec and topology == saved_topology:
                return buffer
        buffer = allocate()
        self._buffers.append((group, mesh, spec, topology, buffer))
        return buffer

    def prepare_inputs(self, host_inputs, mesh, group):
        self.require_preparation()
        inputs = []
        for index, host in enumerate(host_inputs):
            if host is None:
                inputs.append(None)
                continue
            buffer = self._reserve(host, mesh, (group, index), lambda: ttnn.to_device(host, device=mesh))
            ttnn.copy_host_to_device_tensor(host, buffer)
            inputs.append(buffer)
        return tuple(inputs)

    def prepare_output(self, output, group, *, sub_core_grids=None):
        """Allocate/share storage and warm the explicit-destination copy form."""
        self.require_preparation()
        if output is None:
            return None
        if isinstance(output, (tuple, list)):
            return type(output)(
                self.prepare_output(value, (group, i), sub_core_grids=sub_core_grids) for i, value in enumerate(output)
            )
        allocate = (lambda: ttnn.clone(output)) if sub_core_grids is None else (lambda: ttnn.empty_like(output))
        buffer = self._reserve(output, output.device(), group, allocate)
        self.copy_output(output, buffer, sub_core_grids=sub_core_grids)
        return buffer

    @staticmethod
    def copy_output(output, destination, *, sub_core_grids=None):
        """Copy transient model results into prepared storage, including in capture."""
        if output is None and destination is None:
            return None
        if isinstance(destination, (tuple, list)):
            if type(output) is not type(destination) or len(output) != len(destination):
                raise ValueError("Trace output structure changed after preparation")
            for source, target in zip(output, destination):
                PreparedTraceIO.copy_output(source, target, sub_core_grids=sub_core_grids)
        else:
            if (
                output is None
                or destination is None
                or output.spec != destination.spec
                or output.tensor_topology() != destination.tensor_topology()
                or output.device() != destination.device()
            ):
                raise ValueError("Trace output spec changed after preparation")
            kwargs = {} if sub_core_grids is None else {"sub_core_grids": sub_core_grids}
            ttnn.copy(output, destination, **kwargs)
        return destination
