# Functional tests

Place host-side `.cpp` tests here. Each test should compile its own temporary
binary in a `RUN:` line and execute that binary without requiring a device.

Target-only HAL code-generation checks belong under `lit/hal/<architecture>/codegen/`.
