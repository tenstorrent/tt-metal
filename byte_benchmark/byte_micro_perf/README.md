# ByteMicroPerf

## Introduction
micro_perf is a part of ByteMLPerf, which is mainly used to evaluate the performance of frequent computation and communication operators in mainstream deep learning models on new emerging heterogeneous hardwares. The main characteristics are as follows:

- Easy and quick access for diverse heterogeneous hardware
- Evaluation process fitting realistic business scenarios
- Coverage of frequent operators across multiple categories

## Quickstart

### Prepare running environment
```
git clone https://github.com/bytedance/ByteMLPerf.git
cd ByteMLPerf/byte_micro_perf
```

### An example
```
python3 launch.py --hardware_type GPU --task exp
```
#### Usage
```
usage: launch.py [-h] [--hardware_type HARDWARE_TYPE] [--show_hardware_list] [--task_dir TASK_DIR] [--task TASK] [--show_task_list] [--report_dir REPORT_DIR] [--numa_node {-1,0,1}] [--device DEVICE] [--disable_parallel] [--disable_profiling]

options:
  -h, --help            show this help message and exit
  --hardware_type HARDWARE_TYPE
                        The backend going to be evaluted, refs to backends/
  --show_hardware_list  Print all hardware bytemlperf supported
  --task_dir TASK_DIR   The direcotry of tasks going to be evaluted, e.g., default set to workloads
  --task TASK           The task going to be evaluted, refs to workloads/, default use all tasks in workloads/
  --show_task_list      Print all available task names
  --report_dir REPORT_DIR
                        Report dir, default is reports/
  --numa_node {-1,0,1}  NUMA node id, -1 means normal run, default is None which means numa_balance.
  --device DEVICE       Device id, default is all.
  --disable_parallel    Disable parallel run for normal op.
  --disable_profiling   Disable profiling op kernels.
```

### Expected Output
For different types of operators (Compute-bound / Memory-bound), we adopt various metrics to comprehensively evaluate the performance of the operator. Regarding the various metrics, the explanations are as follows:

### for computation ops
| Metric            | Unit          | Description |
| --------          | -------       | ------- |
| latency           | us            | kernel device e2e latency    |
| read_bytes        | B             | bytes read from memory |
| write_bytes       | B             | bytes write to memory |
| io_bytes          | B             | bytes read from memory and write to memory |
| mem_bw            | GB/s          | kernel memory bandwidth |
| calc_flops_power  | TFLOPS / TOPS | testing kernel computing power |
| calc_mem_ratio    | FLOPS / Byte  | algorithm roofline model |

Example:
```
{
    "op_name": "gemm",
    "sku_name": "NVIDIA A800-SXM4-80GB",
    "provider": "default",
    "arguments": {
        "arg_type": "default",
        "dtype": "bfloat16",
        "M": 32768,
        "K": 8192,
        "N": 8192
    },
    "targets": {
        "latency(us)": 14852.47,
        "read_bytes(B)": 671088640,
        "write_bytes(B)": 536870912,
        "io_bytes(B)": 1207959552,
        "mem_bw(GB/s)": 81.331,
        "calc_flops_power(tflops)": 296.116,
        "calc_mem_ratio": 3640.889
    },
    "kernels": [
        "ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_nn"
    ]
}
```


### for communication ops
| Metric            | Unit          | Description |
| --------          | -------       | ------- |
| latency           | us            | kernel device e2e latency    |
| algo size         | B             | algorithm communication size |
| bus size          | B             | bus communication size |
| algo_bw           | GB/s          | algorithm communication bandwidth |
| bus_bw            | GB/s          | bus communication bandwidth |
| latency_list      | list[us]      | latency for each rank |
| algo_bw_list      | list[GB/s]    | algorithm communication bandwidth for each rank |
| bus_bw_list       | list[GB/s]    | bus communication bandwidth for each rank |

Example:
```
{
    "op_name": "all_reduce",
    "sku_name": "NVIDIA A800-SXM4-80GB",
    "provider": "default",
    "arguments": {
        "arg_type": "default",
        "world_size": 8,
        "dtype": "float32",
        "batch_size": 131072,
        "dim_size": 1024
        },
    "targets": {
        "latency(us)": 6281.726,
        "algo_size(B)": 536870912,
        "bus_size(B)": 939524096.0,
        "algo_bw(GB/s)": 85.466,
        "bus_bw(GB/s)": 149.565,
        "algo_bw_sum(GB/s)": 681.339,
        "bus_bw_sum(GB/s)": 1192.343,
        "latency_list(us)": [6299.236, 6314.211, 6312.343, 6311.595, 6304.193, 6301.111, 6305.494, 6281.726], "algo_bw_list(GB/s)": [85.228, 85.026, 85.051, 85.061, 85.161, 85.203, 85.143, 85.466],
        "bus_bw_list(GB/s)": [149.149, 148.795, 148.839, 148.857, 149.032, 149.105, 149.001, 149.565]
    },
    "kernels": [
        "ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevComm*, unsigned long, ncclWork*)"
    ]
}
```

## Trouble Shooting
For more details, you can visit our offical website here: [bytemlperf.ai](https://bytemlperf.ai/). Please let us know if you need any help or have additional questions and issues!
