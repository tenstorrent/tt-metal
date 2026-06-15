# Inner workings of LLK test infra

This document explains, in great detail, what happens when one calls `configuration.run(worker_tensix_coordinates)`. This includes how arguments of the `TestConfig`/`ProfilerConfig` object are translated to C++ code, how kernel executables are generated, and how the kernel itself is executed on a Tensix core on every card Tenstorrent supports.

| | |
|:----|:----|
| 1 | [File and artefact paths](#file-and-artefact-paths) |
| 2 | [TestConfig object](#testconfig-object) |
| 3 | [L1 memory layouts](#l1-memory-layouts) |
| 4 | [Kernel compilation](#kernel-compilation) |
| 5 | [Kernel runtime](#kernel-runtime) |
| 6 | [Data processing](#data-processing) |
---

# File and artefact paths
These static variables of the TestConfig class abstract the usage of file paths used in the testing infrastructure:
```python
class TestConfig:

	# Paths in the tmpfs as to where all build and runtime artefacts end up
	DEFAULT_ARTEFACTS_PATH: ClassVar[Path] = Path("/tmp/tt-llk-build/")
    ARTEFACTS_DIR: ClassVar[Path]
    SHARED_DIR: ClassVar[str]
    SHARED_OBJ_DIR: ClassVar[str]
    SHARED_ELF_DIR: ClassVar[str]
    COVERAGE_INFO_DIR: ClassVar[str]
    SYNC_DIR: ClassVar[Path]

    # C++ code sources directories
    LLK_ROOT: ClassVar[Path]
    TESTS_WORKING_DIR: ClassVar[Path]
    TOOL_PATH: ClassVar[Path]
    HEADER_DIR: ClassVar[Path]

    HELPERS: ClassVar[Path]
    RISCV_SOURCES: ClassVar[Path]
    LINKER_SCRIPTS: ClassVar[Path]

    # Compilation toolchain paths
    GXX: ClassVar[str]
    OBJDUMP: ClassVar[str]
    OBJCOPY: ClassVar[str]
    GCOV: ClassVar[str]
    GCOV_TOOL: ClassVar[str]
```
All of these variables are initialized in `pytest`'s `def pytest_configure(config)` function when `TestConfig.setup_build` is called. Static method `TestConfig.setup_paths` is used to set all aforementioned variables relative to `ARTEFACTS_DIR` of the user's choosing; by default it has the `DEFAULT_ARTEFACTS_PATH` value when initialized.


# TestConfig object
## Build.h generation
Every compiled kernel variant has its own build.h header used to pass Python parameters to the C++ part of the test. That file is located at `/tmp/tt-llk-build/test/path/variant_hash/build.h`. In order to generate this file, the following `TestConfig` constructor arguments are used:
`formats`, `templates`, `runtimes`, `unpack_to_dest`, `disable_format_inference`, `dest_acc`.
Most importantly, `templates` and `runtimes` lists can be populated with object instances of classes defined in `helpers.test_variant_parameters`. These classes wrap types defined in `llk_params.py` and enable C++ code generation.
Elements of the `templates` list generate concrete `constexpr`-like variables used as template arguments to LLK API calls. When the list is processed, the `def convert_to_cpp(self) -> str:` method of every list element is called to generate C++ variable definitions.
Elements of `runtimes` are used to generate the `struct RuntimeParams` declaration by calling `def convert_to_struct_fields(self) -> tuple[str, str]` of each element. The first return tuple element is a struct field type and name, while the second one is how the corresponding `RuntimeParameter`'s `dataclass` field should be serialized when runtime parameters are written to L1. Struct fields in the first argument shall correspond, in the **exact** same order as they are in the dataclass itself. [Here](https://docs.python.org/3/library/struct.html#format-characters) is a detailed list of letters used to specify how Python variables (dataclass fields) should be serialized so C++ code can use them.

## Template and runtime parameters
TODO

# L1 memory layouts
Wormhole and Blackhole cards have the same memory layout. Quasar is similar in every aspect except that it's lacking a BRISC core, and thus space for its elf is not reserved in L1. There are 2 memory layouts for each card - performance and debug. The main difference is that the first utilizes local data memory of RISCV cores, and the latter uses L1 for everything, yielding poorer runtime performance. Layouts are located at `./tt-llk/tests/helpers/ld/*.memory.*.ld`, with performance layouts having nothing at the second wildcard, and debug layouts having `debug`.
L1 specification (capacity, ports, etc.) is [available here](https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/L1.md)


## Variable abstractions
Addresses outlined in the tables below are abstracted away using the following variables:
1. Stimuli space start address &rarr;`StimuliConfig.STIMULI_L1_ADDRESS` in`stimuli_config.py`
2. Runtime arguments struct (C++) &rarr;
	`RUNTIME_PARAMS_TYPE __runtime_args_start[];` in `trisc.cpp`
3. Runtime arguments struct (Python) &rarr; `TestConfig.RUNTIME_ADDRESS` in `test_config.py`
4. TRISC\[0,1,2] perf counter memory &rarr; `ProfilerConfig.THREAD_BUFFER` in `profiler.py`
5. TRISC\[0,1,2] start addresses buffer &rarr; `TestConfig.TRISC_START_ADDRS` in `test_config.py` [1]

## Performance
| Address range | Content |
|---|---|
| 0xFFB00000 - 0xFFB00700 | Local data memory |
| 0xFFB00700 - 0xFFB00800 | Stack |
<center>TRISC L0 performance layout</center>

| Address range | Content |
|---|---|
| 0x00000000 - 0x00003FFF | BRISC code |
| 0x00004000 - 0x00004FFF | BRISC C-runtime data |
| 0x00005000 - 0x00008FFF | TRISC0 (Unpack) code |
| 0x00009000 - 0x000097FF | TRISC0 (Unpack) C-runtime data |
| 0x00009800 - 0x0000D7FF | TRISC1 (Math) code |
| 0x0000D800 - 0x0000DFFF | TRISC1 (Math) C-runtime data |
| 0x0000E000 - 0x00010FFF | TRISC2 (Pack) code |
| 0x00011000 - 0x000117FF | TRISC2 (Pack) C-runtime data |
| 0x00011800 - 0x0001FFFF | Reserved |
| 0x00020000 - 0x0002FFFF | Runtime arguments struct |
| 0x00021000 - 0x00169FFF | Stimuli space |
| 0x0016A000 - 0x0016AFF3 | Performance counters data |
| 0x0016AFF4 - 0x0016AFFF | Profiler barrier |
| 0x0016B000 - 0x0016B3FF | TRISC0 (Unpack) perf counter memory |
| 0x0016C000 - 0x0016C3FF | TRISC1 (Math) perf counter memory |
| 0x0016D000 - 0x0016D3FF | TRISC2 (Pack) perf counter memory |
| 0x0016DFF0 - 0x0016DFFB | TRISC\[0,1,2] start addresses buffer [1] |
<center>Wormhole/Blackhole L1 performance layout</center>


## Debug
| Address range | Content |
|---|---|
| 0xFFB00000 - 0xFFB00800 | Stack |
<center>TRISC L0 debug layout</center>

| Address range | Content |
|---|---|
| 0x00000000 - 0x00007FFF | BRISC code |
| 0x00008000 - 0x0000FFFF | BRISC data memory |
| 0x00010000 - 0x00011FFF | BRISC GCOV memory |
| 0x00012000 - 0x00019FFF | TRISC0 (Unpack) code |
| 0x0001A000 - 0x00021FFF | TRISC0 (Unpack) data memory |
| 0x00022000 - 0x00023FFF | TRISC0 (Unpack) GCOV memory |
| 0x00024000 - 0x0002BFFF | TRISC1 (Math) code |
| 0x0002C000 - 0x00033FFF | TRISC1 (Math) data memory |
| 0x00034000 - 0x00035FFF | TRISC1 (Math) GCOV memory |
| 0x00036000 - 0x0003DFFF | TRISC2 (Pack) code |
| 0x0003E000 - 0x00045FFF | TRISC2 (Pack) data memory |
| 0x00046000 - 0x00047FFF | TRISC2 (Pack) GCOV memory |
| 0x00048000 - 0x00063FFF | Reserved |
| 0x0006E000 - 0x00070000 | Runtime arguments struct |
| 0x00070000 - 0x00169FFF | Stimuli space |
| 0x0016A000 - 0x0016AFF3 | Performance counters data |
| 0x0016AFF4 - 0x0016AFFF | Profiler barrier |
| 0x0016B000 - 0x0016B3FF | TRISC0 (Unpack) perf counter memory |
| 0x0016C000 - 0x0016C3FF | TRISC1 (Math) perf counter memory |
| 0x0016D000 - 0x0016D3FF | TRISC2 (Pack) perf counter memory |
| 0x0016DFF0 - 0x0016DFFB | TRISC\[0,1,2] start addresses buffer [1] |
<center>Wormhole/Blackhole L1 debug layout</center>

# Kernel compilation
This section explains how the TestConfig object, when called, performs compilation of all artifacts necessary to produce `unpack.elf`, `math.elf`, and `pack.elf` files constituting every compiled kernel in our testing infrastructure.

## Firmware
This is the C-Runtime (`void do_crt0(void)` function in `boot.h`), `void main(void)`, and `void gcov_dump(void)` (included only when tests are compiled for coverage) that every RISCV core in every test executes. Its code is located in `trisc.cpp`.
## Functional tests
TODO
## Coverage tests
TODO
## Performance tests
TODO

# Kernel runtime
## Runtime parameters

## Stimuli I/O

## Coverage acquisition

## Performance data acquisition

# Data processing
## Coverage report
## Performance report

\[1] Only used on wormhole
