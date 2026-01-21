# Debug Test for 1x32 Mesh Device All-Gather Issue

This directory contains a C++ debug program to investigate why `all_gather` is using the physical mesh shape (4x8 or 8x4) instead of the logical mesh shape (1x32).

## Files

- `test_1x32_all_gather_debug.cpp` - C++ test program that mimics the Python test
- `build_debug_test.sh` - Build script
- `CMakeLists.txt` - CMake configuration

## Building

The test is automatically included in the main CMake build. To build it:

```bash
cd build_Debug
cmake --build . --target test_1x32_all_gather_debug -j$(nproc)
```

Or use the build script:
```bash
./test-run/build_debug_test.sh
```

## Running with GDB

```bash
cd build_Debug
gdb test-run/test_1x32_all_gather_debug
```

### Key Breakpoints to Set

1. **After mesh device creation** - Check if shape is correct:
   ```
   break test_1x32_all_gather_debug.cpp:34
   ```

2. **Inside all_gather** - Check what mesh_shape is being used:
   ```
   break ttnn/cpp/ttnn/operations/ccl/all_gather/all_gather.cpp:36
   ```

3. **In get_boundary_mode** - Check what shape is used there:
   ```
   break ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:39
   ```

### Useful GDB Commands

```gdb
# Print mesh device shape
print mesh_device->shape()

# Print mesh device view shape
print mesh_device->get_view().shape()

# Print tensor device shape
print ones_tensor.device()->shape()

# Print tensor device view shape
print ones_tensor.device()->get_view().shape()

# Step into all_gather to see what shape it uses
break ttnn/cpp/ttnn/operations/ccl/all_gather/all_gather.cpp:36
continue
print mesh_shape
```

## Debug Points in the Code

The program has 4 debug points that print mesh shapes:

1. **DEBUG POINT 1** - Right after mesh device creation
2. **DEBUG POINT 2** - After tensor creation
3. **DEBUG POINT 3** - Right before calling all_gather
4. **DEBUG POINT 4** - After all_gather call

## Expected Output

If everything is correct, you should see:
```
Requested mesh shape: [1, 32]
DEBUG POINT 1 - mesh_device->shape(): [1, 32]
DEBUG POINT 1 - mesh_device->get_view().shape(): [1, 32]
DEBUG POINT 2 - tensor.device()->shape(): [1, 32]
DEBUG POINT 2 - tensor.device()->get_view().shape(): [1, 32]
DEBUG POINT 3 - About to call all_gather
  Input tensor device shape: [1, 32]
  Input tensor device view shape: [1, 32]
```

If there's a bug, you might see `[4, 8]` or `[8, 4]` instead of `[1, 32]` at some point.

## Investigation Steps

1. Run the program and check all debug points
2. Set breakpoint at `all_gather.cpp:36` and check what `mesh_shape` is
3. Check if `device()->shape()` differs from `device()->get_view().shape()`
4. Trace back to see where the shape gets corrupted
