# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from tests.sweep_framework.sweep_utils.conv_transpose2d_common import run_short
import ttnn
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import (
    get_mesh_shape,
    create_mesh_device,
)
from tests.sweep_framework.master_config_loader_v2 import MasterConfigLoader
from tests.sweep_framework.sweep_utils.op_kwargs_utils import build_op_kwargs

# Override the default timeout in seconds for hang detection.
TIMEOUT = 300

# Load traced configurations from real model tests (V2 format)
loader = MasterConfigLoader()
model_traced_params = loader.get_suite_parameters("conv_transpose2d")

parameters = {
    "model_traced_sample": {
        "input_specs": [
            # [batch_size, input_channels, input_height, input_width, output_channels,
            #  kernel_height, kernel_width, stride_h, stride_w, pad_h, pad_w,
            #  dilation_h, dilation_w, out_pad_h, out_pad_w]
            (1, 128, 16, 16, 128, 2, 2, 2, 2, 0, 0, 1, 1, 0, 0),
        ],
        "storage_type": ["StorageType::DEVICE"],
    },
}

if model_traced_params:
    parameters["model_traced"] = model_traced_params


def mesh_device_fixture():
    mesh_shape = get_mesh_shape()
    if mesh_shape:
        try:
            device = create_mesh_device(mesh_shape)
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_mesh_device(device)
        except Exception as e:
            print(f"Failed to create mesh device {mesh_shape}: {e}, falling back to single device")
            device = ttnn.open_device(device_id=0, l1_small_size=32768, dispatch_core_config=ttnn.DispatchCoreConfig())
            device_name = ttnn.get_arch_name()
            yield (device, device_name)
            ttnn.close_device(device)
    else:
        # conv_transpose2d requires L1 small buffer allocation
        device = ttnn.open_device(device_id=0, l1_small_size=32768, dispatch_core_config=ttnn.DispatchCoreConfig())
        device_name = ttnn.get_arch_name()
        yield (device, device_name)
        ttnn.close_device(device)
        del device


def run(
    input_specs=None,
    compute_config=None,
    dtype=None,
    storage_type="StorageType::DEVICE",
    *,
    device,
    **kwargs,
) -> list:
    # Build input_specs from flat kwargs when not provided directly
    if input_specs is None:
        batch_size = kwargs.get("batch_size")
        out_channels = kwargs.get("out_channels")
        in_channels = kwargs.get("in_channels")
        input_height = kwargs.get("input_height") or kwargs.get("input_h")
        input_width = kwargs.get("input_width") or kwargs.get("input_w")
        kernel_size = kwargs.get("kernel_size")
        stride = kwargs.get("stride")
        padding = kwargs.get("padding")
        dilation = kwargs.get("dilation")
        output_padding = kwargs.get("output_padding")

        if batch_size is not None and out_channels is not None and in_channels is not None:
            kh, kw = (
                (kernel_size if isinstance(kernel_size, (list, tuple)) else [kernel_size, kernel_size])
                if kernel_size
                else [2, 2]
            )
            sh, sw = (stride if isinstance(stride, (list, tuple)) else [stride, stride]) if stride else [1, 1]
            ph, pw = (padding if isinstance(padding, (list, tuple)) else [padding, padding]) if padding else [0, 0]
            dh, dw = (dilation if isinstance(dilation, (list, tuple)) else [dilation, dilation]) if dilation else [1, 1]
            oph, opw = (
                (output_padding if isinstance(output_padding, (list, tuple)) else [output_padding, output_padding])
                if output_padding
                else [0, 0]
            )
            ih = input_height or 4
            iw = input_width or 4
            input_specs = (
                int(batch_size),
                int(in_channels),
                int(ih),
                int(iw),
                int(out_channels),
                int(kh),
                int(kw),
                int(sh),
                int(sw),
                int(ph),
                int(pw),
                int(dh),
                int(dw),
                int(oph),
                int(opw),
            )
        else:
            return [(False, "Cannot construct input_specs: missing batch_size/out_channels/in_channels"), 0.0]

    result = run_short(list(input_specs), device)

    pcc_passed = bool(result[0])
    pcc_value = str(result[1])
    e2e_perf = result[2]

    return [(pcc_passed, pcc_value), e2e_perf]
