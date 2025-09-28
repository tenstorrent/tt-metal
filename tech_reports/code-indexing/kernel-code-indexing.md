# Kernel code indexing

Host code is built with the help of CMake, so it's easy to generate `compile_commands.json` that can be used by some code indexing plugins to provide things like "Go to definition", "Find references", and code highlighting.

Add `-e` or `--export-compile-commands` to `./build_metal.sh` to generate `compile_commands.json` in the build directory for the host code.

Kernels are compiled and linked in runtime with `runtime/sfpi/compiler/bin/riscv32-tt-elf-g++`, and they naturally aren't part of the CMake project, so code indexing doesn't work there.

## 1. Create a fake CMake target to enable kernel code indexing
Add `--enable-fake-kernels-target` to `build_metal.sh`

- This approach doesn't work very well:
  - many kernels depend on different defines and compile-time arguments
  - There is a compute kernel split into unpacker/packer/math
  - jit build generates some files at runtime and includes many different files implicitly
  - includes depend on the architecture
- But still, many IDE features should work, especially inside a kernel file


## 2. Generate kernel `compile_commands.json` with `bear` utility
Another way to achieve code indexing in kernels is using the bear tool, which can generate compile_commands.json from compilation logs.

This approach generates a more precise compile_commands.json, since it parses flags and definitions passed during the actual compilation of the kernel. But at the same time, it's a bit more involved.

### Prerequisites
- bear (`sudo apt-get install bear`)
- python 3.10+

### Steps
1. `export TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1`
2. Prepare a test/program that calls a kernel(s) of interest. Example: `./experiments/calls_ttnn_mean.py`

**./experiments/calls_ttnn_mean.py**
```python
import ttnn
import torch

device = ttnn.open_device(device_id=0)

def call_mean(device):
    torch.manual_seed(42)
    shape = (2, 1, 64, 64)
    torch_input = torch.rand(shape).bfloat16()
    ttnn_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    ttnn_mean = ttnn.mean(ttnn_input, dim=0, keepdim=True)
    ttnn_mean_torch = ttnn.to_torch(ttnn_mean)
    print("TTNN Mean Result:", ttnn_mean_torch)

try:
    call_mean(device)
finally:
    ttnn.close_device(device)
```

This will generate something like:
```
...

2025-07-22 17:08:31.374 | info     |    BuildKernels |     g++ link cmd: cd /home/pasha/.cache/tt-metal-cache/35b1b61bc1/4097/kernels/reduce_h/16191301163007683913/trisc1/ && /home/pasha/projects/tt-metal/runtime/sfpi/compiler/bin/riscv32-tt-elf-g++ -O3 -Wl,--just-symbols=/home/pasha/.cache/tt-metal-cache/35b1b61bc1/4097/firmware/trisc1/trisc1_weakened.elf -mcpu=tt-wh -std=c++17 -flto=auto -ffast-math -g -fno-exceptions -Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -nostartfiles -T/home/pasha/projects/tt-metal/runtime/hw/toolchain/wormhole/kernel_trisc1.ld -Wl,--emit-relocs trisck.o /home/pasha/projects/tt-metal/runtime/hw/lib/wormhole/substitutes.o -o /home/pasha/.cache/tt-metal-cache/35b1b61bc1/4097/kernels/reduce_h/16191301163007683913/trisc1/trisc1.elf (build.cpp:825)
...
```

3. Run the script that would run bear, and properly update the project's `compile_commands.json`:

`python3 ./scripts/build_kernel_compile_commands_json.py --input-command="python3 /home/ubuntu/projects/tt-metal/experiments/calls_ttnn_mean.py" --output-dir="build_Debug" --merge`

### build_kernel_compile_commands_json.py scrip

Due to the nuances of how kernels are built on the device, there are a few things to keep in mind:
1. All kernels for NCRISC/TRISC0/TRISC1/TRISC2/BRISC are included in ncrisc.cc/trisc0.cc/trisc1.cc/trisc2.cc/brisc.cc, so to enable indexing inside the kernel, the script has to search for actual kernel files and update generated compile_commands.json.
2. Code for each TRISC0/TRISC1/TRISC2 is built from the same kernel, with the help of some macros. Currently, the script just removes all duplicates, so a `compile_commands.json` will be generated for a random one.

- **Note!!! All paths in the --input-command must be absolute, since it is being called inside the temporary directory**

```
usage: build_kernel_compile_commands_json.py [-h] --input-command INPUT_COMMAND --output-dir OUTPUT_DIR [--choose-first-kernel-file] [--overwrite] [--merge] [--search-dir SEARCH_DIR]

Process compile_commands.json for TT Metal kernels

options:
  -h, --help            show this help message and exit
  --input-command INPUT_COMMAND
                        The command to run with a bear
  --output-dir OUTPUT_DIR
                        Directory to store the updated compile_commands.json
  --choose-first-kernel-file
                        Automatically choose the first kernel file found
  --overwrite           Overwrite existing compile_commands.json without asking
  --merge               Merge with existing compile_commands.json without asking
  --search-dir SEARCH_DIR
                        Directory to search for kernel files (default: current directory)
```


## Using `compile_commands.json`
### VS Code:
- `clangd` by default searches for compile_commands.json in the project root, but it can be changed. `settings.json` example:
```json
    "clangd.path": "/usr/bin/clangd-17",
    "clangd.arguments": [
        "-background-index",
        "-pretty",
        "-compile-commands-dir=${workspaceFolder}/build_Debug"
    ],
```
- Microsoft Intellisense extension can work with compile commands if configured properly:
```json
    "C_Cpp.default.compileCommands": "${workspaceFolder}/build_Debug/compile_commands.json",
```
