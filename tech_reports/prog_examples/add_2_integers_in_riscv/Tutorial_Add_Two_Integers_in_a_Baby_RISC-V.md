# Tutorial - Add Two Integers in a Baby RISC-V 🚧
1. To build and execute, you may use the following commands:
```bash
    export TT_METAL_HOME=$(pwd)
    ./build_metal.sh --build-programming-examples
    ./build/programming_examples/metal_example_add_2_integers_in_riscv
```
2. Setup the host program:

```Device *device = CreateDevice(0);
CommandQueue& cq = device->command_queue();
Program program = CreateProgram();
constexpr CoreCoord core = {0, 0};
```
3. Create two integers:

```
Int a=1
Int b=2
```

4. Setup the kernel function for integer addition:

```
KernelHandle binary_reader_kernel_id = CreateKernel(
    program,
    "tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp",
    core,
    DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
```
```
Kernel Code Here - TBD
```
5. Setup two integers as runtime arguments:

```
EnqueProgram(cq, program, false);
Finish(cq);
```

6. Close the device:

```
CloseDevice(device);
```
