# Fabric Router Pause/Resume Usage Guide

## Table of Contents

- [Introduction](#introduction)
- [Quick Start](#quick-start)
- [Common Use Cases](#common-use-cases)
- [Complete Examples](#complete-examples)
- [Best Practices](#best-practices)
- [Testing Patterns](#testing-patterns)
- [Debugging Guide](#debugging-guide)
- [Integration Checklist](#integration-checklist)

## Introduction

This guide provides practical examples and patterns for using the fabric router pause/resume functionality in your tests and applications. It focuses on real-world usage scenarios and best practices.

### Who Should Read This

- Test developers writing fabric-related tests
- Engineers implementing fabric control features
- Developers debugging fabric routing issues
- Anyone needing to control traffic flow in TT-Metal fabric

### Prerequisites

Before using the pause/resume APIs, ensure:

- Fabric topology is configured and initialized
- At least 2 devices available in fabric
- Telemetry is enabled in firmware build
- FabricRouterStateManager is available

## Quick Start

### Minimal Pause/Resume Example

```cpp
#include <gtest/gtest.h>
#include "tests/tt_metal/tt_fabric/common/fabric_fixture.hpp"
#include "tests/tt_metal/tt_fabric/common/fabric_command_interface.hpp"

TEST_F(Fabric1DFixture, SimplePauseTest) {
    // Get control plane
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // Create command interface
    test_utils::FabricCommandInterface cmd_interface(control_plane);

    // Pause all routers
    cmd_interface.pause_routers();

    // Wait for pause to complete (5 second timeout)
    bool success = cmd_interface.wait_for_pause();
    ASSERT_TRUE(success) << "Failed to pause routers";

    // Verify all routers are paused
    ASSERT_TRUE(cmd_interface.all_routers_in_state(RouterStateCommon::PAUSED));

    log_info(LogTest, "All routers successfully paused");
}
```

### Basic Pause and Resume Cycle

```cpp
TEST_F(Fabric1DFixture, PauseResumeTest) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    test_utils::FabricCommandInterface cmd_interface(control_plane);

    // Pause
    cmd_interface.pause_routers();
    ASSERT_TRUE(cmd_interface.wait_for_pause());

    // Resume
    cmd_interface.resume_routers();
    bool resumed = cmd_interface.wait_for_state(RouterStateCommon::RUNNING);
    ASSERT_TRUE(resumed) << "Failed to resume routers";

    log_info(LogTest, "Pause/resume cycle completed");
}
```

## Common Use Cases

### Use Case 1: Pause Before Reconfiguration

When reconfiguring fabric, pause all traffic first:

```cpp
void ReconfigureFabricSafely() {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    test_utils::FabricCommandInterface cmd_interface(control_plane);

    // Step 1: Pause all traffic
    log_info(LogFabric, "Pausing fabric traffic for reconfiguration");
    cmd_interface.pause_routers();

    if (!cmd_interface.wait_for_pause()) {
        throw std::runtime_error("Failed to pause fabric");
    }

    // Step 2: Perform reconfiguration
    log_info(LogFabric, "Fabric paused, performing reconfiguration");
    // ... reconfiguration code here ...

    // Step 3: Resume traffic
    log_info(LogFabric, "Reconfiguration complete, resuming traffic");
    cmd_interface.resume_routers();

    if (!cmd_interface.wait_for_state(RouterStateCommon::RUNNING)) {
        throw std::runtime_error("Failed to resume fabric");
    }

    log_info(LogFabric, "Fabric reconfiguration complete");
}
```

### Use Case 2: Traffic Validation in Tests

Validate traffic before and after operations:

```cpp
TEST_F(Fabric1DFixture, ValidateTrafficDuringPause) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto mesh_ids = control_plane.get_user_physical_mesh_ids();
    ASSERT_FALSE(mesh_ids.empty());
    MeshId mesh_id = mesh_ids[0];
    size_t num_devices = get_devices().size();

    // Launch traffic generators (application-specific)
    LaunchTrafficGenerators();

    // Validate traffic is initially flowing
    bool flowing = test_utils::validate_traffic_flowing(
        control_plane, mesh_id, num_devices);
    ASSERT_TRUE(flowing) << "No traffic detected initially";

    // Pause routers
    test_utils::FabricCommandInterface cmd_interface(control_plane);
    cmd_interface.pause_routers();
    ASSERT_TRUE(cmd_interface.wait_for_pause());

    // Validate traffic stopped
    bool stopped = test_utils::validate_traffic_stopped(
        control_plane, mesh_id, num_devices);
    ASSERT_TRUE(stopped) << "Traffic still flowing after pause";

    // Resume routers
    cmd_interface.resume_routers();
    ASSERT_TRUE(cmd_interface.wait_for_state(RouterStateCommon::RUNNING));

    // Validate traffic resumes
    flowing = test_utils::validate_traffic_flowing(
        control_plane, mesh_id, num_devices);
    ASSERT_TRUE(flowing) << "Traffic did not resume";

    CleanupTrafficGenerators();
}
```

### Use Case 3: Debugging with State Inspection

Use observability utilities to debug issues:

```cpp
void DebugFabricState() {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto mesh_ids = control_plane.get_user_physical_mesh_ids();

    // Log detailed router states
    log_info(LogTest, "=== Current Fabric State ===");
    test_utils::log_all_router_states(control_plane, mesh_ids);

    // Get aggregate counts
    auto state_counts = test_utils::count_routers_by_state(control_plane, mesh_ids);

    log_info(LogTest, "State Summary:");
    for (const auto& [state, count] : state_counts) {
        log_info(LogTest, "  {}: {} routers",
            test_utils::router_state_to_string(state), count);
    }

    // Check specific router
    FabricNodeId suspicious_node{.mesh_id = 0, .logical_x = 1, .logical_y = 0};
    chan_id_t channel = 0;

    test_utils::FabricCommandInterface cmd_interface(control_plane);
    RouterStateCommon state = cmd_interface.get_router_state(suspicious_node, channel);

    log_info(LogTest, "Suspicious router state: {}",
        test_utils::router_state_to_string(state));
}
```

### Use Case 4: Timeout Handling

Handle timeouts gracefully:

```cpp
TEST_F(Fabric1DFixture, HandlePauseTimeout) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    test_utils::FabricCommandInterface cmd_interface(control_plane);

    cmd_interface.pause_routers();

    auto start = std::chrono::steady_clock::now();
    bool paused = cmd_interface.wait_for_pause(std::chrono::milliseconds(5000));
    auto duration = std::chrono::steady_clock::now() - start;

    if (paused) {
        log_info(LogTest, "Pause completed in {} ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(duration).count());
    } else {
        log_warning(LogTest, "Pause timeout after {} ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(duration).count());

        // Debug which routers failed to pause
        auto mesh_ids = control_plane.get_user_physical_mesh_ids();
        test_utils::log_all_router_states(control_plane, mesh_ids);

        // Decide how to proceed
        GTEST_SKIP() << "Pause timeout - hardware issue suspected";
    }
}
```

### Use Case 5: Conditional Operations Based on State

Check state before operations:

```cpp
void SafeFabricOperation() {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    test_utils::FabricCommandInterface cmd_interface(control_plane);

    // Check if routers are already paused
    if (cmd_interface.all_routers_in_state(RouterStateCommon::PAUSED)) {
        log_info(LogTest, "Routers already paused, skipping pause");
    } else {
        log_info(LogTest, "Routers running, issuing pause");
        cmd_interface.pause_routers();
        if (!cmd_interface.wait_for_pause()) {
            throw std::runtime_error("Failed to pause fabric");
        }
    }

    // Perform operation that requires paused fabric
    PerformSensitiveOperation();

    // Resume if we paused
    if (cmd_interface.all_routers_in_state(RouterStateCommon::PAUSED)) {
        cmd_interface.resume_routers();
        cmd_interface.wait_for_state(RouterStateCommon::RUNNING);
    }
}
```

## Complete Examples

### Example 1: Full Test with Traffic Generators

```cpp
#include <gtest/gtest.h>
#include <chrono>
#include <thread>
#include "tests/tt_metal/tt_fabric/common/fabric_fixture.hpp"
#include "tests/tt_metal/tt_fabric/common/fabric_command_interface.hpp"
#include "tests/tt_metal/tt_fabric/common/fabric_traffic_validation.hpp"
#include "tests/tt_metal/tt_fabric/common/fabric_worker_kernel_helpers.hpp"

class MyFabricTest : public Fabric1DFixture {
protected:
    void SetUp() override {
        Fabric1DFixture::SetUp();
        workers_launched_ = false;
    }

    void TearDown() override {
        if (workers_launched_) {
            CleanupWorkers();
        }
        Fabric1DFixture::TearDown();
    }

    void LaunchWorkers() {
        auto devices = get_devices();
        for (size_t i = 0; i < devices.size() - 1; ++i) {
            auto& device = devices[i];
            auto& dest_device = devices[i + 1];

            // Allocate memory
            auto mem_layout = test_utils::allocate_worker_memory(device);
            worker_mem_layouts_.push_back(mem_layout);

            // Create kernel
            CoreCoord core{0, 0};
            FabricNodeId dest = test_utils::get_fabric_node_id(dest_device);
            auto program = test_utils::create_traffic_generator_program(
                device, core, dest, mem_layout);

            // Launch
            RunProgramNonblocking(device, *program);

            worker_programs_.push_back(program);
            worker_cores_.push_back(core);
        }
        workers_launched_ = true;
    }

    void CleanupWorkers() {
        auto devices = get_devices();
        for (size_t i = 0; i < worker_programs_.size(); ++i) {
            test_utils::signal_worker_teardown(
                devices[i], worker_cores_[i],
                worker_mem_layouts_[i].teardown_signal_address);
        }
        for (size_t i = 0; i < worker_programs_.size(); ++i) {
            try {
                test_utils::wait_for_worker_complete(
                    this, devices[i], *worker_programs_[i],
                    std::chrono::milliseconds(1000));
            } catch (const std::exception& e) {
                log_error(LogTest, "Worker {} timeout: {}", i, e.what());
            }
        }
        workers_launched_ = false;
        worker_programs_.clear();
        worker_mem_layouts_.clear();
        worker_cores_.clear();
    }

private:
    bool workers_launched_;
    std::vector<std::shared_ptr<tt_metal::Program>> worker_programs_;
    std::vector<test_utils::WorkerMemoryLayout> worker_mem_layouts_;
    std::vector<CoreCoord> worker_cores_;
};

TEST_F(MyFabricTest, CompleteWorkflow) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto mesh_ids = control_plane.get_user_physical_mesh_ids();
    ASSERT_FALSE(mesh_ids.empty());
    MeshId mesh_id = mesh_ids[0];
    size_t num_devices = get_devices().size();
    ASSERT_GE(num_devices, 2);

    // Launch traffic
    LaunchWorkers();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Validate traffic
    bool flowing = test_utils::validate_traffic_flowing(
        control_plane, mesh_id, num_devices);
    ASSERT_TRUE(flowing);

    // Pause
    test_utils::FabricCommandInterface cmd_interface(control_plane);
    cmd_interface.pause_routers();
    ASSERT_TRUE(cmd_interface.wait_for_pause());

    // Verify stopped
    bool stopped = test_utils::validate_traffic_stopped(
        control_plane, mesh_id, num_devices);
    ASSERT_TRUE(stopped);

    // Resume
    cmd_interface.resume_routers();
    ASSERT_TRUE(cmd_interface.wait_for_state(RouterStateCommon::RUNNING));

    // Verify resumed
    flowing = test_utils::validate_traffic_flowing(
        control_plane, mesh_id, num_devices);
    ASSERT_TRUE(flowing);

    CleanupWorkers();
}
```

### Example 2: Performance Measurement Test

```cpp
TEST_F(Fabric1DFixture, MeasurePauseLatency) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    test_utils::FabricCommandInterface cmd_interface(control_plane);

    std::vector<int64_t> pause_latencies;
    std::vector<int64_t> resume_latencies;

    constexpr int num_iterations = 10;

    for (int i = 0; i < num_iterations; ++i) {
        // Measure pause
        auto start = std::chrono::steady_clock::now();
        cmd_interface.pause_routers();
        bool paused = cmd_interface.wait_for_pause();
        auto pause_duration = std::chrono::steady_clock::now() - start;
        ASSERT_TRUE(paused);

        int64_t pause_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            pause_duration).count();
        pause_latencies.push_back(pause_ms);

        // Brief pause
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Measure resume
        start = std::chrono::steady_clock::now();
        cmd_interface.resume_routers();
        bool resumed = cmd_interface.wait_for_state(RouterStateCommon::RUNNING);
        auto resume_duration = std::chrono::steady_clock::now() - start;
        ASSERT_TRUE(resumed);

        int64_t resume_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            resume_duration).count();
        resume_latencies.push_back(resume_ms);

        log_info(LogTest, "Iteration {}: Pause {}ms, Resume {}ms",
            i, pause_ms, resume_ms);
    }

    // Calculate statistics
    auto avg_pause = std::accumulate(pause_latencies.begin(),
        pause_latencies.end(), 0LL) / num_iterations;
    auto avg_resume = std::accumulate(resume_latencies.begin(),
        resume_latencies.end(), 0LL) / num_iterations;

    log_info(LogTest, "=== Performance Summary ===");
    log_info(LogTest, "Average Pause Latency: {} ms", avg_pause);
    log_info(LogTest, "Average Resume Latency: {} ms", avg_resume);

    // Verify performance requirements (NFR-1)
    EXPECT_LT(avg_pause, 500) << "Average pause latency exceeds 500ms";
    EXPECT_LT(avg_resume, 500) << "Average resume latency exceeds 500ms";
}
```

### Example 3: Multi-Mesh Testing

```cpp
TEST_F(Fabric1DFixture, MultiMeshPauseResume) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto all_mesh_ids = control_plane.get_user_physical_mesh_ids();

    if (all_mesh_ids.empty()) {
        GTEST_SKIP() << "No meshes available";
    }

    log_info(LogTest, "Testing {} meshes", all_mesh_ids.size());

    test_utils::FabricCommandInterface cmd_interface(control_plane);

    // Log initial state
    log_info(LogTest, "Initial state:");
    test_utils::log_all_router_states(control_plane, all_mesh_ids);

    // Pause all meshes
    cmd_interface.pause_routers();
    bool paused = cmd_interface.wait_for_pause();
    ASSERT_TRUE(paused) << "Failed to pause all meshes";

    // Verify all meshes paused
    auto state_counts = test_utils::count_routers_by_state(control_plane, all_mesh_ids);
    log_info(LogTest, "After pause:");
    for (const auto& [state, count] : state_counts) {
        log_info(LogTest, "  {}: {}",
            test_utils::router_state_to_string(state), count);
    }

    EXPECT_EQ(state_counts[RouterStateCommon::RUNNING], 0)
        << "Some routers still running";
    EXPECT_GT(state_counts[RouterStateCommon::PAUSED], 0)
        << "No routers paused";

    // Resume all meshes
    cmd_interface.resume_routers();
    bool resumed = cmd_interface.wait_for_state(RouterStateCommon::RUNNING);
    ASSERT_TRUE(resumed) << "Failed to resume all meshes";

    // Verify all meshes running
    state_counts = test_utils::count_routers_by_state(control_plane, all_mesh_ids);
    log_info(LogTest, "After resume:");
    for (const auto& [state, count] : state_counts) {
        log_info(LogTest, "  {}: {}",
            test_utils::router_state_to_string(state), count);
    }

    EXPECT_GT(state_counts[RouterStateCommon::RUNNING], 0)
        << "No routers running";
    EXPECT_EQ(state_counts[RouterStateCommon::PAUSED], 0)
        << "Some routers still paused";
}
```

## Best Practices

### 1. Always Use RAII for Worker Cleanup

```cpp
// GOOD: RAII ensures cleanup
class MyTest : public Fabric1DFixture {
protected:
    ~MyTest() {
        if (workers_launched_) {
            CleanupWorkers();
        }
    }
};

// BAD: Manual cleanup can be forgotten
TEST_F(Fabric1DFixture, BadTest) {
    LaunchWorkers();
    // ... test code ...
    // Forgot to call CleanupWorkers()!
}
```

### 2. Check Return Values

```cpp
// GOOD: Check return values
if (!cmd_interface.wait_for_pause()) {
    log_error(LogTest, "Pause failed");
    test_utils::log_all_router_states(control_plane, mesh_ids);
    FAIL() << "Pause timeout";
}

// BAD: Ignore return values
cmd_interface.wait_for_pause();  // What if it times out?
```

### 3. Use Appropriate Timeouts

```cpp
// GOOD: Customize timeout based on operation
bool paused = cmd_interface.wait_for_pause(
    std::chrono::milliseconds(10000));  // Longer for slow hardware

// BAD: Default timeout may be too short
bool paused = cmd_interface.wait_for_pause();  // 5s may not be enough
```

### 4. Log State for Debugging

```cpp
// GOOD: Log state before and after
log_info(LogTest, "Before pause:");
test_utils::log_all_router_states(control_plane, mesh_ids);

cmd_interface.pause_routers();
cmd_interface.wait_for_pause();

log_info(LogTest, "After pause:");
test_utils::log_all_router_states(control_plane, mesh_ids);

// BAD: No debugging information
cmd_interface.pause_routers();
cmd_interface.wait_for_pause();
```

### 5. Stabilize Before Validation

```cpp
// GOOD: Allow traffic to stabilize
LaunchWorkers();
std::this_thread::sleep_for(std::chrono::milliseconds(500));
bool flowing = validate_traffic_flowing(...);

// BAD: Validate immediately
LaunchWorkers();
bool flowing = validate_traffic_flowing(...);  // May fail spuriously
```

### 6. Use Longer Sample Intervals for Stopped Traffic

```cpp
// GOOD: Longer interval for stopped traffic validation
bool stopped = test_utils::validate_traffic_stopped(
    control_plane, mesh_id, num_devices,
    std::chrono::milliseconds(200));  // More confidence

// ACCEPTABLE: Default interval
bool stopped = test_utils::validate_traffic_stopped(
    control_plane, mesh_id, num_devices);  // 100ms default
```

## Testing Patterns

### Pattern 1: State Transition Test

```cpp
void TestStateTransition(RouterStateCommon from, RouterStateCommon to) {
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    test_utils::FabricCommandInterface cmd_interface(control_plane);

    // Ensure starting state
    if (!cmd_interface.all_routers_in_state(from)) {
        // Transition to starting state
        if (from == RouterStateCommon::PAUSED) {
            cmd_interface.pause_routers();
            cmd_interface.wait_for_pause();
        } else {
            cmd_interface.resume_routers();
            cmd_interface.wait_for_state(RouterStateCommon::RUNNING);
        }
    }

    // Perform transition
    if (to == RouterStateCommon::PAUSED) {
        cmd_interface.pause_routers();
        ASSERT_TRUE(cmd_interface.wait_for_pause());
    } else {
        cmd_interface.resume_routers();
        ASSERT_TRUE(cmd_interface.wait_for_state(RouterStateCommon::RUNNING));
    }

    // Verify final state
    ASSERT_TRUE(cmd_interface.all_routers_in_state(to));
}

TEST_F(Fabric1DFixture, AllStateTransitions) {
    TestStateTransition(RouterStateCommon::RUNNING, RouterStateCommon::PAUSED);
    TestStateTransition(RouterStateCommon::PAUSED, RouterStateCommon::RUNNING);
    TestStateTransition(RouterStateCommon::RUNNING, RouterStateCommon::RUNNING);  // No-op
    TestStateTransition(RouterStateCommon::PAUSED, RouterStateCommon::PAUSED);    // No-op
}
```

### Pattern 2: Parameterized Timeout Tests

```cpp
class TimeoutTest : public Fabric1DFixture,
                   public ::testing::WithParamInterface<int> {};

TEST_P(TimeoutTest, CustomTimeout) {
    int timeout_ms = GetParam();
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    test_utils::FabricCommandInterface cmd_interface(control_plane);

    cmd_interface.pause_routers();
    bool paused = cmd_interface.wait_for_pause(
        std::chrono::milliseconds(timeout_ms));

    if (paused) {
        log_info(LogTest, "Pause succeeded with {}ms timeout", timeout_ms);
    } else {
        log_info(LogTest, "Pause timed out at {}ms", timeout_ms);
    }
}

INSTANTIATE_TEST_SUITE_P(
    TimeoutValues,
    TimeoutTest,
    ::testing::Values(100, 500, 1000, 5000, 10000));
```

### Pattern 3: Fixture with Automatic Traffic

```cpp
class TrafficEnabledTest : public Fabric1DFixture {
protected:
    void SetUp() override {
        Fabric1DFixture::SetUp();
        if (get_devices().size() >= 2) {
            LaunchWorkers();
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    void TearDown() override {
        if (workers_launched_) {
            CleanupWorkers();
        }
        Fabric1DFixture::TearDown();
    }

    // Worker management methods...
};

TEST_F(TrafficEnabledTest, PauseWithActiveTraffic) {
    // Traffic already flowing from SetUp
    auto& cp = tt::tt_metal::MetalContext::instance().get_control_plane();

    bool flowing = test_utils::validate_traffic_flowing(cp, 0, get_devices().size());
    ASSERT_TRUE(flowing);

    // Test pause...
}
```

## Debugging Guide

### Problem: Pause Timeout

**Symptoms:**
- `wait_for_pause()` returns false
- Test fails with "Routers did not enter PAUSED state within timeout"

**Debug Steps:**

1. **Check which routers failed:**
```cpp
if (!cmd_interface.wait_for_pause()) {
    log_error(LogTest, "Pause timeout - inspecting router states:");
    test_utils::log_all_router_states(control_plane, mesh_ids);

    auto counts = test_utils::count_routers_by_state(control_plane, mesh_ids);
    log_error(LogTest, "State counts:");
    for (const auto& [state, count] : counts) {
        log_error(LogTest, "  {}: {}",
            test_utils::router_state_to_string(state), count);
    }
}
```

2. **Check individual router states:**
```cpp
auto router_cores = cmd_interface.get_all_router_cores();
for (const auto& [node, channel] : router_cores) {
    auto state = cmd_interface.get_router_state(node, channel);
    log_info(LogTest, "Router ({}, {}, {}) channel {}: {}",
        node.mesh_id, node.logical_x, node.logical_y, channel,
        test_utils::router_state_to_string(state));
}
```

3. **Increase timeout:**
```cpp
// Try longer timeout
bool paused = cmd_interface.wait_for_pause(std::chrono::milliseconds(15000));
```

### Problem: Traffic Not Detected

**Symptoms:**
- `validate_traffic_flowing()` returns false
- Test fails with "Traffic not detected before pause command"

**Debug Steps:**

1. **Check workers launched:**
```cpp
log_info(LogTest, "Workers launched: {}", workers_launched_);
log_info(LogTest, "Num programs: {}", worker_programs_.size());
```

2. **Increase sample interval:**
```cpp
bool flowing = test_utils::validate_traffic_flowing(
    control_plane, mesh_id, num_devices,
    std::chrono::milliseconds(500));  // Longer sample
```

3. **Check telemetry manually:**
```cpp
auto snap1 = test_utils::capture_telemetry_snapshot(cp, mesh_id, num_devices);
std::this_thread::sleep_for(std::chrono::milliseconds(500));
auto snap2 = test_utils::capture_telemetry_snapshot(cp, mesh_id, num_devices);

for (const auto& [node, channels] : snap1.words_sent_per_channel) {
    for (const auto& [channel, count1] : channels) {
        uint64_t count2 = snap2.words_sent_per_channel[node][channel];
        log_info(LogTest, "Node ({},{},{}) Ch {}: {} -> {} (delta: {})",
            node.mesh_id, node.logical_x, node.logical_y, channel,
            count1, count2, count2 - count1);
    }
}
```

### Problem: Traffic Still Flowing After Pause

**Symptoms:**
- `validate_traffic_stopped()` returns false after pause
- Test fails with "Traffic detected during PAUSED state"

**Debug Steps:**

1. **Verify pause completed:**
```cpp
bool paused = cmd_interface.wait_for_pause();
ASSERT_TRUE(paused) << "Pause didn't complete";

// Double-check all routers paused
ASSERT_TRUE(cmd_interface.all_routers_in_state(RouterStateCommon::PAUSED))
    << "Not all routers paused";
```

2. **Add delay before validation:**
```cpp
cmd_interface.wait_for_pause();
std::this_thread::sleep_for(std::chrono::milliseconds(200));  // Let in-flight packets complete
bool stopped = test_utils::validate_traffic_stopped(...);
```

3. **Check for rogue traffic sources:**
```cpp
// Verify workers were paused before checking traffic
CleanupWorkers();  // Stop all workers
std::this_thread::sleep_for(std::chrono::milliseconds(100));
bool stopped = test_utils::validate_traffic_stopped(...);
```

## Integration Checklist

When integrating pause/resume into your code:

- [ ] Include all necessary headers
- [ ] Extend Fabric1DFixture or create equivalent fixture
- [ ] Implement RAII cleanup for workers
- [ ] Check return values from all API calls
- [ ] Add appropriate timeouts for your hardware
- [ ] Log state transitions for debugging
- [ ] Handle timeout cases gracefully
- [ ] Add stabilization delays where appropriate
- [ ] Test on minimum configuration (2 devices)
- [ ] Test on larger configurations (4+ devices)
- [ ] Verify telemetry is enabled
- [ ] Document any custom timeout values used
- [ ] Add appropriate assertions with clear messages
- [ ] Consider thread safety (use single thread)
- [ ] Profile pause/resume latency if performance-critical

## Summary

The fabric router pause/resume functionality provides powerful control over fabric traffic. Follow these key principles:

1. **Always check return values** - Timeouts can happen
2. **Use observability** - Log states for debugging
3. **Handle cleanup properly** - Use RAII patterns
4. **Allow stabilization time** - Traffic needs time to start/stop
5. **Test thoroughly** - Different configurations behave differently

For more information, see:
- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Architecture](ARCHITECTURE.md) - System design details
- [Main README](README_PAUSE_RESUME.md) - Feature overview
