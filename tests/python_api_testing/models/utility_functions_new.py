import time
import tt_lib
import torch
import numpy as np
from loguru import logger
from tt_lib.utils import _nearest_32
from os import environ


def is_close(a, b, rtol=1e-2, atol=1e-2, max_mag=2.0, max_mag_fraction=0.02):
    """
    A variant of np.isclose with logging.
    """
    absdiff = (a - b).abs()
    reldiff1 = (a.abs() / b.abs()) - 1.0
    reldiff2 = (a.abs() + 1.0) / (b.abs() + 1.0) - 1.0  # in case b.abs() is 0
    reldiff_or = torch.logical_or(reldiff1.abs() < rtol, reldiff2.abs() < rtol)
    max_mag_ok = absdiff < max_mag * max_mag_fraction

    or_abs_rel = torch.logical_or(absdiff < atol, reldiff_or)
    or_abs_rel = torch.logical_or(or_abs_rel, max_mag_ok)
    debug_index = or_abs_rel.to(torch.int32).argmin().item()

    if not or_abs_rel.reshape(-1)[debug_index]:
        logger.info("isclose mismatch at index=", debug_index)
        logger.info(a.reshape(-1)[debug_index])
        logger.info(b.reshape(-1)[debug_index])
        logger.info("reldiff1=", reldiff1.reshape(-1)[debug_index])
        logger.info("reldiff2=", reldiff2.reshape(-1)[debug_index])
        logger.info("absdiff=", absdiff.reshape(-1)[debug_index])

        HT = a.shape[-2] // 32
        WT = a.shape[-1] // 32
        hwt = debug_index//1024
        wt = hwt % WT
        ht = hwt // WT
        h = (debug_index % 1024) // 32
        w = (debug_index % 1024) % 32

        logger.info("****    at ", debug_index, " --- ", "HTWT=", ht, wt, "HW=", h, w)

    return torch.all(or_abs_rel)


def get_oom_of_float(float_lst):
    """
    Given a list of floats, returns a list of the order or magnitudes
    of the floats. Useful when you want to make sure that even if your
    tt outputs don't match pytorch all that well, they are at least
    on the same order of magnitude
    """
    ooms = []
    for el in float_lst:
        str_el = str(el)
        if "e" in str_el:
            oom = int(str_el.split("e")[1])
        elif str_el[:2] == "0.":
            str_el = str_el.split(".")[1]

            oom = -1
            for e in str_el:
                if e != "0":
                    break
                oom -= 1
        else:
            oom = len(str_el.split(".")[0])

        ooms.append(oom)

    return ooms


def enable_persistent_kernel_cache():
    """
    Enables persistent compiled kernel caching - disables recompiling the kernels for the duration of running process if built_kernels/.../hash directory with kernel binaries is present.
    """
    tt_lib.device.EnablePersistentKernelCache()


def disable_persistent_kernel_cache():
    """
    Disables persistent compiled kernel caching. This is the default state.
    """
    tt_lib.device.DisablePersistentKernelCache()

def enable_compilation_reports():
    """
    Enables generating reports of compilation statistics in .reports/tt_metal dir
    """
    return tt_lib.device.EnableCompilationReports()


def disable_compilation_reports():
    """
    Disables generating reports of compilation statistics
    """
    return tt_lib.device.DisableCompilationReports()


def enable_memory_reports():
    """
    Enables generating reports of memory allocation statistics in .reports/tt_metal dir
    """
    return tt_lib.device.EnableMemoryReports()


def disable_memory_reports():
    """
    Disables generating reports of memory allocation statistics
    """
    return tt_lib.device.DisableMemoryReports()


def comp_allclose(golden, calculated, rtol=1e-05, atol=1e-08):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    atol_delta = torch.max(torch.abs(golden - calculated)).item()
    rtol_delta = torch.max(
        torch.abs(golden - calculated) / torch.abs(calculated)
    ).item()
    return (
        torch.allclose(golden, calculated, rtol, atol, True),
        f"Max ATOL Delta: {atol_delta}, Max RTOL Delta: {rtol_delta}",
    )


def comp_pcc(golden, calculated, pcc=0.99):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    if torch.all(torch.isnan(golden)) and torch.all(torch.isnan(calculated)):
        logger.warning("Both tensors are 'nan'")
        return True, f"PCC: {1.0}"

    if torch.all(torch.isnan(golden)) or torch.all(torch.isnan(calculated)):
        logger.error("One tensor is all nan, the other is not.")
        return False, f"PCC: {0.0}"

    # Test if either is completely zero
    if torch.any(golden.bool()) != torch.any(calculated.bool()):
        logger.error("One tensor is all zero")
        return False, f"PCC: {0.0}"

    # if torch.any(torch.isinf(golden)) or torch.any(torch.isinf(calculated)):
    #    raise RuntimeError(f"Tensor overflow to infinity: \n{golden}\n{calculated}")

    # if torch.any(torch.isneginf(golden)) or torch.any(torch.isneginf(calculated)):
    #    raise RuntimeError(f"Tensor overflow to negative infinity: \n{golden}\n{calculated}")

    # For now, mask all infs and nans so that we check the rest... TODO
    golden = golden.clone()
    golden[
        torch.logical_or(
            torch.isnan(golden),
            torch.logical_or(torch.isinf(golden), torch.isneginf(golden)),
        )
    ] = 0
    calculated = calculated.clone()
    calculated[
        torch.logical_or(
            torch.isnan(calculated),
            torch.logical_or(torch.isinf(calculated), torch.isneginf(calculated)),
        )
    ] = 0

    if torch.equal(golden, calculated):
        return True, f"PCC: {1.0}"

    if golden.dtype == torch.bfloat16:
        golden = golden.type(torch.float32)
        calculated = calculated.type(torch.float32)
    cal_pcc = np.min(
        np.ma.corrcoef(
            np.ma.masked_invalid(torch.squeeze(golden).detach().numpy()).flatten(),
            np.ma.masked_invalid(torch.squeeze(calculated).detach().numpy()).flatten(),
        )
    )

    if isinstance(cal_pcc, np.ma.core.MaskedConstant):
        return True, f"PCC: {1.0}"

    return cal_pcc >= pcc, f"PCC: {cal_pcc}"


def comp_allclose_and_pcc(golden, calculated, rtol=1e-05, atol=1e-08, pcc=0.99):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    passing = True
    output = ""
    passing_allclose, output_allclose = comp_allclose(golden, calculated, rtol, atol)
    passing &= passing_allclose
    output += output_allclose
    if torch.numel(golden) != 1:
        passing_pcc, output_pcc = comp_pcc(golden, calculated, pcc)
        passing &= passing_pcc
        output += f", {output_pcc}"

    return passing, output


def torch2tt_tensor(py_tensor: torch.Tensor, tt_device, tt_layout=tt_lib.tensor.Layout.TILE, tt_memory_config=tt_lib.tensor.MemoryConfig(True)):
    size = list(py_tensor.size())

    while len(size) < 4:
        size.insert(0, 1)

    py_tensor = py_tensor.reshape(size)
    tt_tensor = tt_lib.tensor.Tensor(py_tensor, tt_lib.tensor.DataType.BFLOAT16).to(tt_layout)

    tt_tensor = tt_tensor.to(tt_device, tt_memory_config)

    return tt_tensor


def tt2torch_tensor(tt_tensor):
    tt_output = tt_tensor.cpu()
    if tt_output.layout() != tt_lib.tensor.Layout.ROW_MAJOR:
        tt_output = tt_output.to(tt_lib.tensor.Layout.ROW_MAJOR)
    return tt_output.to_torch()


def pad_by_zero(x: torch.Tensor, device):
    initial_shape = x.shape
    if initial_shape[3] % 32 != 0 or initial_shape[2] % 32 != 0:
        x = tt_lib.tensor.Tensor(x, tt_lib.tensor.DataType.BFLOAT16)
        x = x.pad((initial_shape[0], initial_shape[1], _nearest_32(initial_shape[2]), _nearest_32(initial_shape[3])), (0, 0, 0, 0), 0)
        x = x.to(tt_lib.tensor.Layout.TILE).to(device)

    else:
        x = torch2tt_tensor(x, device)
    return x, initial_shape


def unpad_from_zero(x, desired_shape):
    if x.shape()[-1] == desired_shape[-1] and x.shape()[-2] == desired_shape[-2] :
        x = tt2torch_tensor(x)
    else:
        x = x.cpu()
        if(x.layout() != tt_lib.tensor.Layout.ROW_MAJOR):
            x = x.to(tt_lib.tensor.Layout.ROW_MAJOR)
        x = x.unpad((0, 0, 0, 0), (desired_shape[0] - 1, desired_shape[1] - 1, desired_shape[2] - 1, desired_shape[3] - 1) )
        x = x.to_torch()
    return x


class Profiler():
    def __init__(self):
        self.start_times = dict()
        self.times = dict()
        self.disabled = False

    def enable(self):
        self.disabled = False

    def disable(self):
        self.disabled = True

    def start(self, key):
        if self.disabled:
            return

        self.start_times[key] = time.time()

    def end(self, key, PERF_CNT=1):
        if self.disabled:
            return

        if key not in self.start_times:
            return

        diff = time.time() - self.start_times[key]

        if key not in self.times:
            self.times[key] = []

        self.times[key].append(diff / PERF_CNT)

    def get(self, key):
        if key not in self.times:
            return 0

        return sum(self.times[key]) / len(self.times[key])

    def print(self):
        for key in self.times:
            average = self.get(key)
            logger.info(f"{key}: {average:.3f}s")


profiler = Profiler()


def tt_to_torch_tensor(tt_tensor):
    tt_output = tt_tensor.cpu().to(tt_lib.tensor.Layout.ROW_MAJOR)
    return tt_output.to_torch()


def torch_to_tt_tensor_rm(py_tensor, device, shape=None, put_on_device=True):
    if shape is None:
        shape = list(py_tensor.size())
        while len(shape) < 4:
            shape.insert(0, 1)

    tt_tensor = (
         tt_lib.tensor.Tensor(py_tensor.reshape(shape), tt_lib.tensor.DataType.BFLOAT16)
    )
    if put_on_device:
        tt_tensor = tt_tensor.to(device)
    return tt_tensor


def torch_to_tt_tensor(py_tensor, device):
    shape = list(py_tensor.size())
    while len(shape) < 4:
        shape.insert(0, 1)

    tt_tensor = (
         tt_lib.tensor.Tensor(py_tensor.reshape(shape), tt_lib.tensor.DataType.BFLOAT16)
        .to(tt_lib.tensor.Layout.TILE)     # change memory layout of TT Tensor to TILE (as operation that will use it expects TILE layout)
        .to(device)                         # move TT Tensor from host to TT accelerator device (device is of type tt_lib.device.Device)
    )

    return tt_tensor

def prep_report(model_name: str, batch_size: int, inference_and_compile_time: float, inference_time: float, expected_compile_time: float, expected_inference_time: float, comments: str, inference_time_cpu: float=None):
    today = time.strftime("%Y_%m_%d")

    def write_dict_to_file(csv_path, dict_res):
        columns = ", ".join([str(d) for d in dict_res.keys()])
        # values = ", ".join([("{:.2f}".format(d) if isinstance(d, float) else str(d)) for d in dict_res.values()])
        values = ", ".join([d for d in dict_res.values()])

        with open(csv_path, "w") as csvfile:
            csvfile.write(columns)
            csvfile.write("\n")
            csvfile.write(values)


    compile_time = inference_and_compile_time - inference_time
    gs_throughput = "{:.4f}".format(batch_size * (1/inference_time))
    cpu_throughput = batch_size * (1/inference_time_cpu) if inference_time_cpu else "unknown"
    cpu_throughput = "{:.4f}".format(cpu_throughput) if not isinstance(cpu_throughput, str) else cpu_throughput
    dict_res = {
        "Model": model_name,
        "Setting": comments,
        "Batch": str(batch_size),
        "First Run (sec)": "{:.2f}".format(inference_and_compile_time),
        "Second Run (sec)":  "{:.2f}".format(inference_time),
        "Compile Time (sec)": "{:.2f}".format(compile_time),
        "Expected Compile Time (sec)": "{:.2f}".format(expected_compile_time),
        "Inference Time GS (sec)": "{:.4f}".format(inference_time),
        "Expected Inference Time GS (sec)": "{:.4f}".format(expected_inference_time),
        "Throughput GS (batch*inf/sec)": gs_throughput,
        "Inference Time CPU (sec)": "{:.4f}".format(inference_time_cpu),
        "Throughput CPU (batch*inf/sec)": cpu_throughput,
    }

    csv_file = f"perf_{model_name}_{today}.csv"
    write_dict_to_file(csv_file, dict_res)
