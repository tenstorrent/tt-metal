In order to keep our test suite clean, organized and searchable, please follow the guidelines provided below when adding new tests, modifying existing tests or deleting outdated tests.

<!-- toc -->

Table of Contents
=================

- [Table of Contents](#table-of-contents)
  - [Test Naming](#test-naming)
  - [Fixture Naming](#fixture-naming)
  - [Fixture Organization](#fixture-organization)
  - [File Naming](#file-naming)
  - [File Organization](#fixture-organization)
    - [api/](#api)
    - [debug_tools/](#debug_tools)
    - [device/](#device)
    - [dispatch/](#dispatch)
    - [eth/](#eth)
    - [integration/](#integration)
    - [llk/](#llk)
    - [stl/](#stl)
    - [test_kernels/](#test_kernels)
    - [common/](#common)

<!-- Created by https://luciopaiva.com/markdown-toc/ -->

<!-- tocstop -->

## Test Naming
Prefix test names with the core type(s) that the test is using.
 - If it's using Tensix cores, prefix it with `Tensix`
 - If it's using active ethernet cores, prefix it with `ActiveEth`
 - If it's using idle ethernet cores, prefix it with `IdleEth`
 - If it's using both active and idle ethernet cores, prefix it with `Eth`
 - If it's using multiple core types, prefix it with each core type, eg. `TensixActiveEth`, `TensixIdleEth`, `TensixEth`, etc.
 - If it isn't using any core type, don't prefix it with anything

## Fixture Naming
 - All fixture names should end in `Fixture`

## Fixture Organization
Before creating a new fixture, check if an existing fixture meets your needs. If you need to create a new fixture, consider subclassing an existing fixture to avoid duplicating functionality already provided by another fixture.

## File Naming
 - Files that contain fixtures should have their names end with `_fixture`
 - Files that contain helper functions and/or test utilities should have their names end with `_test_utils`
 - Files that contain tests should have their names start with `test_`

## File Organization
Place test utility files and fixture files as close as possible to the files that rely on them. For example, if you have a test file `test_A.cpp` in `tests/tt_metal/tt_metal/dispatch/dispatch_buffer/` and another test file `test_B.cpp` in `tests/tt_metal/tt_metal/dispatch/dispatch_program/`, and both need to use a fixture file `C_fixture.hpp`, it is logical to place `C_fixture.hpp` in `tests/tt_metal/tt_metal/dispatch/`. This ensures the fixture is easily accessible to the relevant test files while avoiding unnecessary clutter in a more generic directory like `tests/tt_metal/tt_metal/common/`.

Tests using Google Test should be placed in one of the directories listed below that best aligns with their purpose. If multiple directories seem suitable, use your best judgment to select the most appropriate one.

__Important note: only tests that use Google Test should be placed in the following directories.__

### `api/`
 - Contains tests that explicitly test `tt-metal`'s API

### `debug_tools/`
 - Contains tests for DPrint and Watcher

### `device/`
 - Contains tests for device initialization and teardown
 - Contains tests that check device-specific properties
 - Contains tests that read from and/or write to the device, but don't launch anything

### `dispatch/`
 - Contains tests that explicitly test for properties relating to dispatch
 - Contains both slow dispatch and fast dispatch tests

### `eth/`
 - Contains tests for ethernet communication between 2 or more devices

### `integration/`
 - Contains tests for real-world use cases, eg. matmul, etc

### `llk/`
 -

### `stl/`
 - Contains tests which test custom data structures and algorithms used in `tt-metal`
 - None of the tests in this directory should run on the device

The following directories should be reserved for files that support testing but should not contain actual tests themselves.

### `test_kernels/`
 - Contains kernels that are used in tests

### `common/`
 - Contains test fixtures and utilities shared across multiple directories listed above
