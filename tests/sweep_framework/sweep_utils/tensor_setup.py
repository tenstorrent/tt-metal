# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Keep sweep tensor setup out of the device-perf profiler window.

gather_single_test_perf() sums every program dispatched since the previous profiler read,
so a device program launched while a sweep builds its inputs is charged to the op under
test. Two setup paths dispatch programs:

- ``ttnn.from_torch(..., device=mesh_device, mesh_mapper=...)`` constructs on the device: a
  row-major write, then tilize and, when the torch dtype differs, typecast
  (``convert_python_tensor_to_tt_tensor``, ttnn/core/tensor/py_to_tt_tensor.cpp, mesh branch
  only; the single-device branch builds on host and only writes). The sweep paths that hit
  it are ``mesh_tensor_utils.replicate_with_topology``, ``mesh_tensor_utils.create_tensor_on_mesh``
  and every module that passes ``mesh_mapper`` itself.
- ``ttnn.to_memory_config`` on a device tensor is a full copy. The two mesh helpers use it to
  reshard a DRAM-interleaved tensor to the traced sharded config, and fifteen model_traced
  modules reshard inputs the same way before their op.

While device perf is requested both are routed through the host: build with ``from_torch``
without a device and write with ``to_device``, or read back with ``from_device`` and write
again. Neither dispatches a program. The result has the same dtype, layout and memory_config;
for from_torch-produced interleaved sources the whole TensorSpec matches
(test_tensor_setup_device.py). ``enable_bfloat_opt`` only affects device-side construction and
is a no-op on this path.

``from_torch`` is never the op under test, so it is rerouted for the whole run().
``to_memory_config`` can be the op (interleaved_to_sharded_e2e) or part of it
(global_avg_pool2d converts sharded inputs internally), so it is rerouted only outside the
module's start_measuring_time() / stop_measuring_time() bracket. Inside the bracket everything
runs on the device and is counted; outside it is setup or teardown and is not. A module without
the bracket has to_memory_config rerouted throughout and can set ``_DEVICE_SIDE_SETUP = True``
to keep device-side construction; ``--device-side-setup`` (``TTNN_SWEEP_DEVICE_SIDE_SETUP=1``)
does the same for a whole run.

Known asymmetries. e2e_perf measured inside the bracket includes a host round trip where it
used to include a device program, for modules that build tensors there. Peak-memory capture
(memory_utils) runs the module with device-side setup, since readback under NO_DISPATCH is
undefined, so perf and memory rows for one vector describe different setup paths.
``--trace-params`` records the rerouted calls (from_torch without a device, then to_device)
and the pre-op hooks fire for those inner calls, not for the module's own.
"""

import contextlib
import functools

from framework.sweeps_logger import sweeps_logger as logger
from tests.ttnn import utils_for_testing

_in_measured_window = False
_announced = False


def _window_listener(entered: bool) -> None:
    global _in_measured_window
    _in_measured_window = entered


def setup_context(test_module, config):
    """The context execute_test() runs a module under: host-side setup when device perf is
    requested and neither the run nor the module opted out, otherwise nothing."""
    if not getattr(config, "measure_device_perf", False):
        return contextlib.nullcontext()
    if getattr(config, "device_side_setup", False) or getattr(test_module, "_DEVICE_SIDE_SETUP", False):
        return contextlib.nullcontext()
    return host_side_tensor_construction()


def _named_like(wrapper, original):
    # Only the naming attributes. functools.wraps would also copy the registered ttnn
    # Operation's __dict__ onto a plain function that then half-impersonates it.
    functools.update_wrapper(
        wrapper, original, assigned=("__module__", "__name__", "__qualname__", "__doc__"), updated=()
    )
    return wrapper


@contextlib.contextmanager
def host_side_tensor_construction():
    import ttnn

    global _announced
    orig_from_torch = ttnn.from_torch
    orig_to_memory_config = ttnn.to_memory_config

    def _from_torch(tensor, dtype=None, **kwargs):
        device = kwargs.pop("device", None)
        if tensor is None or device is None:
            return orig_from_torch(tensor, dtype, **kwargs)
        memory_config = kwargs.pop("memory_config", None)
        cq_id = kwargs.pop("cq_id", None)
        spec = kwargs.get("spec")
        if spec is not None and memory_config is None:
            # from_torch takes the placement from the spec; to_device needs it spelled out.
            memory_config = spec.memory_config
        elif spec is not None:
            # from_torch rejects memory_config alongside spec; let it raise as it does unpatched.
            kwargs["memory_config"] = memory_config
        host = orig_from_torch(tensor, dtype, **kwargs)
        # from_torch spells the queue kwarg cq_id; to_device spells it queue_id and accepts None.
        return ttnn.to_device(host, device, memory_config=memory_config, queue_id=cq_id)

    def _to_memory_config(tensor, memory_config, dtype=None, **kwargs):
        if _in_measured_window:
            # Inside start/stop_measuring_time this is the op under test, or part of it.
            return orig_to_memory_config(tensor, memory_config, dtype, **kwargs)
        if dtype is not None:
            # A dtype change is a typecast; a host round trip cannot reproduce it.
            return orig_to_memory_config(tensor, memory_config, dtype, **kwargs)
        if kwargs:
            # output_tensor= is a preallocated device buffer the round trip would leave untouched.
            return orig_to_memory_config(tensor, memory_config, dtype, **kwargs)
        if not ttnn.is_tensor_storage_on_device(tensor):
            return orig_to_memory_config(tensor, memory_config, dtype, **kwargs)
        device = tensor.device()
        host = ttnn.from_device(tensor)
        return ttnn.to_device(host, device, memory_config=memory_config)

    ttnn.from_torch = _named_like(_from_torch, orig_from_torch)
    ttnn.to_memory_config = _named_like(_to_memory_config, orig_to_memory_config)
    utils_for_testing._measuring_window_listeners.append(_window_listener)
    if not _announced:
        logger.info(
            "Device perf: sweep inputs are built on the host; only programs inside "
            "start/stop_measuring_time count toward the op (--device-side-setup to disable)"
        )
        _announced = True
    try:
        yield
    finally:
        ttnn.from_torch = orig_from_torch
        ttnn.to_memory_config = orig_to_memory_config
        utils_for_testing._measuring_window_listeners.remove(_window_listener)
        _window_listener(False)
