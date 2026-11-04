import time, os, sys
import torch, ttnn

DM = '#include "api/dataflow/dataflow_api.h"\nvoid kernel_main() {}\n'
CP = '#include "api/compute/common.h"\nvoid kernel_main() {}\n'


def desc(cores, core_list, n_args, n_common, out):
    kernels = []
    for i in range(3):
        rt = ttnn.RuntimeArgs()
        for c in core_list:
            rt[c.x][c.y] = list(range(n_args))
        cfg = (
            ttnn.ComputeConfigDescriptor()
            if i == 2
            else ttnn.DataMovementConfigDescriptor(
                processor=[ttnn.DataMovementProcessor.RISCV_0, ttnn.DataMovementProcessor.RISCV_1][i],
                noc=[ttnn.NOC.NOC_0, ttnn.NOC.NOC_1][i],
            )
        )
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=CP if i == 2 else DM,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                compile_time_args=[n_args, len(core_list)],
                runtime_args=rt,
                common_runtime_args=list(range(n_common)),
                config=cfg,
            )
        )
    cbs = [
        ttnn.CBDescriptor(
            total_size=4096,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=j, data_format=ttnn.bfloat16, page_size=2048)],
        )
        for j in range(9)
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


device = ttnn.open_device(device_id=0, l1_small_size=0)
try:
    out = ttnn.from_torch(torch.zeros(32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    for ncores in (110, 32, 1):
        core_list = ttnn.corerange_to_cores(
            ttnn.num_cores_to_corerangeset(ncores, ttnn.CoreCoord(11, 10), True), None, True
        )
        cores = ttnn.num_cores_to_corerangeset(ncores, ttnn.CoreCoord(11, 10), True)
        for n_args, n_common in ((0, 4), (4, 4), (8, 4), (26, 4), (0, 224)):
            d = desc(cores, core_list, n_args, n_common, out)
            ttnn.generic_op([out], d)
            ttnn.synchronize_device(device)
            ts = []
            for _ in range(100):
                t0 = time.perf_counter()
                ttnn.generic_op([out], d)
                ts.append(time.perf_counter() - t0)
                ttnn.synchronize_device(device)
            ts.sort()
            print(f"FLOOR cores={ncores} per_core_args={n_args} common={n_common}: {1e6*ts[50]:.1f} us")
finally:
    ttnn.close_device(device)
