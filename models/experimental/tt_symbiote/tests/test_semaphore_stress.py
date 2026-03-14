# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Semaphore stress tests designed to EXPOSE semaphore issues in traced mode.

TDD-style tests - these are designed to FAIL and expose the semaphore
contention/drift issues in traced async CCL operations. Once the underlying
semaphore management is fixed, these tests should pass.

Expected failure modes:
1. Timeout/hang due to semaphore value drift over many replays
2. Semaphore contention between multiple layers sharing the same ccl_manager
3. Async operation corruption from rapid-fire replays without sync
4. Numerical correctness drift from reduce_scatter producing wrong partial results

Run with:
    MESH_DEVICE=T3K TT_SYMBIOTE_RUN_MODE=TRACED pytest models/experimental/tt_symbiote/tests/test_semaphore_stress.py -v --timeout=120
"""

import os
import pytest
import torch
import torch.nn as nn
import ttnn

from models.experimental.tt_symbiote.core.run_config import TracedRun, disable_trace
from models.experimental.tt_symbiote.modules.linear import TTNNLinearIColShardedWRowSharded
from models.experimental.tt_symbiote.utils.device_management import set_device


def _is_traced_mode():
    """Check if traced mode is active."""
    return os.environ.get("TT_SYMBIOTE_RUN_MODE") == "TRACED"


def _is_t3k():
    """Check if T3K mesh device is configured."""
    return os.environ.get("MESH_DEVICE") == "T3K"


# T3K device mesh mapping
MESH_DEVICE_MAP = {
    "N150": (1, 1),
    "N300": (1, 2),
    "N150x4": (1, 4),
    "T3K": (1, 8),
    "TG": (8, 4),
    "P150": (1, 1),
    "P300": (1, 2),
    "P150x4": (1, 4),
    "P150x8": (1, 8),
    "BHGLX": (8, 4),
}

# T3K device parameters for traced mode
T3K_DEVICE_PARAMS = {
    "l1_small_size": 245760,
    "trace_region_size": 50000000,
    "num_command_queues": 2,
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
}


def create_sharded_linear(mesh_device, name_suffix="", in_features=None, out_features=None):
    """Helper to create a sharded linear layer with proper setup.

    Args:
        mesh_device: The mesh device to use.
        name_suffix: Suffix for the layer's unique name.
        in_features: Input features. Defaults to 256 * num_devices if None.
        out_features: Output features. Defaults to 256 if None.
    """
    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1

    # Create dimensions that work with sharding
    if in_features is None:
        in_features = 256 * num_devices  # e.g., 2048 for 8 devices
    if out_features is None:
        out_features = 256

    # Create PyTorch reference layer
    torch_linear = nn.Linear(in_features, out_features, bias=False)
    torch_linear = torch_linear.to(torch.bfloat16)

    # Create TTNN sharded linear
    ttnn_linear = TTNNLinearIColShardedWRowSharded.from_torch(torch_linear)
    ttnn_linear._unique_name = f"stress_test_{name_suffix}"
    set_device(ttnn_linear, mesh_device, dump_visualization=False)
    ttnn_linear.preprocess_weights()
    ttnn_linear.move_weights_to_device()

    return ttnn_linear, torch_linear, in_features


@pytest.mark.timeout(60)
@pytest.mark.skipif(not _is_t3k(), reason="Requires MESH_DEVICE=T3K")
@pytest.mark.parametrize("device_params", [T3K_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize("run_mode", ["NON_TRACED", "TRACED"])
def test_semaphore_exhaustion_many_replays(mesh_device, run_mode):
    """
    Test designed to EXPOSE semaphore exhaustion over many trace replays.

    Hypothesis: Semaphore values drift after many replays without proper reset,
    eventually causing a hang/timeout when semaphore values wrap around or
    exceed expected bounds.

    Expected failure: Hang/timeout after N replays where N depends on
    semaphore pool size and cycling logic.
    """
    TracedRun.release_all()

    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    if num_devices < 2:
        pytest.skip(f"Need at least 2 devices for sharded test, got {num_devices}")

    # Helper function for non-traced execution
    @disable_trace
    def run_nontraced(layer, input_tensor):
        return layer(input_tensor)

    ttnn_linear, _, in_features = create_sharded_linear(mesh_device, "exhaustion")
    input_shape = (1, 32, in_features)

    # Phase 1: Capture trace
    print(f"\n[Exhaustion Test - {run_mode}] Phase 1: Capturing trace with {num_devices} devices...")
    input1 = torch.randn(*input_shape, dtype=torch.bfloat16)
    _ = ttnn_linear(input1)
    print(f"  Trace capture complete. Cache size: {TracedRun.cache_size()}")

    # Phase 2: Run 100 replays without explicit synchronization
    # This tests if semaphore values properly cycle/reset
    num_replays = 100
    print(f"\n[Exhaustion Test - {run_mode}] Phase 2: Running {num_replays} replays without sync...")

    for i in range(num_replays):
        input_new = torch.randn(*input_shape, dtype=torch.bfloat16)

        if (i + 1) % 10 == 0:
            print(f"  Replay {i+1}/{num_replays}...")

        # This may hang if semaphore cycling is broken
        if run_mode == "NON_TRACED":
            output = run_nontraced(ttnn_linear, input_new)
        else:
            output = ttnn_linear(input_new)

        assert output is not None, f"Replay {i+1} returned None"

    print(f"\n[Exhaustion Test - {run_mode}] SUCCESS: All {num_replays} replays completed!")
    TracedRun.release_all()


@pytest.mark.timeout(60)
@pytest.mark.skipif(not _is_t3k(), reason="Requires MESH_DEVICE=T3K")
@pytest.mark.parametrize("device_params", [T3K_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize("run_mode", ["NON_TRACED", "TRACED"])
def test_semaphore_contention_multiple_layers(mesh_device, run_mode):
    """
    Test designed to EXPOSE semaphore contention between multiple layers.

    Hypothesis: Multiple sharded layers sharing the same ccl_manager can
    have semaphore contention when their traces are interleaved, because
    each layer's trace was captured with a specific semaphore state that
    may not match reality when replayed in a different order.

    Expected failure: Hang or incorrect results when interleaving replays
    from multiple layers.
    """
    TracedRun.release_all()

    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    if num_devices < 2:
        pytest.skip(f"Need at least 2 devices for sharded test, got {num_devices}")

    # Helper function for non-traced execution
    @disable_trace
    def run_nontraced(layer, input_tensor):
        return layer(input_tensor)

    # Create 4 separate sharded linear layers sharing the same mesh_device
    # (and thus the same ccl_manager via device_state)
    num_layers = 4
    layers = []
    layer_inputs = []

    print(f"\n[Contention Test - {run_mode}] Creating {num_layers} sharded layers...")
    for i in range(num_layers):
        ttnn_linear, _, in_features = create_sharded_linear(mesh_device, f"layer_{i}")
        layers.append(ttnn_linear)
        input_shape = (1, 32, in_features)
        layer_inputs.append(input_shape)

    # Phase 1: Capture traces for all layers
    print(f"\n[Contention Test - {run_mode}] Phase 1: Capturing traces for all {num_layers} layers...")
    for i, (layer, input_shape) in enumerate(zip(layers, layer_inputs)):
        input_tensor = torch.randn(*input_shape, dtype=torch.bfloat16)
        _ = layer(input_tensor)
        print(f"  Layer {i} trace captured")

    print(f"  Total trace cache size: {TracedRun.cache_size()}")

    # Phase 2: Interleave replays from all layers
    # This tests if semaphore state is properly isolated per-trace
    num_rounds = 25
    print(f"\n[Contention Test - {run_mode}] Phase 2: Interleaving {num_rounds} rounds of replays...")

    for round_idx in range(num_rounds):
        if (round_idx + 1) % 5 == 0:
            print(f"  Round {round_idx+1}/{num_rounds}...")

        # Interleave: run each layer once per round
        for layer_idx, (layer, input_shape) in enumerate(zip(layers, layer_inputs)):
            input_tensor = torch.randn(*input_shape, dtype=torch.bfloat16)

            # This may hang or produce incorrect results due to semaphore contention
            if run_mode == "NON_TRACED":
                output = run_nontraced(layer, input_tensor)
            else:
                output = layer(input_tensor)

            assert output is not None, f"Round {round_idx+1}, Layer {layer_idx} returned None"

    print(f"\n[Contention Test - {run_mode}] SUCCESS: All {num_rounds} rounds completed!")
    TracedRun.release_all()


@pytest.mark.timeout(60)
@pytest.mark.skipif(not _is_t3k(), reason="Requires MESH_DEVICE=T3K")
@pytest.mark.parametrize("device_params", [T3K_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize("run_mode", ["NON_TRACED", "TRACED"])
def test_semaphore_rapid_fire_no_sync(mesh_device, run_mode):
    """
    Test designed to EXPOSE async operation corruption from rapid-fire replays.

    Hypothesis: Issuing many trace replays rapidly without synchronization
    can cause async CCL operations to overlap in unexpected ways, corrupting
    results or causing hangs.

    Expected failure: Incorrect output values or hang.
    """
    TracedRun.release_all()

    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    if num_devices < 2:
        pytest.skip(f"Need at least 2 devices for sharded test, got {num_devices}")

    # Helper function for non-traced execution
    @disable_trace
    def run_nontraced(layer, input_tensor):
        return layer(input_tensor)

    ttnn_linear, torch_linear, in_features = create_sharded_linear(mesh_device, "rapid_fire")
    input_shape = (1, 32, in_features)

    # Phase 1: Capture trace
    print(f"\n[Rapid Fire Test - {run_mode}] Phase 1: Capturing trace...")
    input1 = torch.randn(*input_shape, dtype=torch.bfloat16)
    _ = ttnn_linear(input1)
    print(f"  Trace capture complete")

    # Phase 2: Rapid-fire 50 replays, save outputs
    num_replays = 50
    print(f"\n[Rapid Fire Test - {run_mode}] Phase 2: Issuing {num_replays} rapid-fire replays...")

    outputs = []
    inputs = []

    for i in range(num_replays):
        input_tensor = torch.randn(*input_shape, dtype=torch.bfloat16)
        inputs.append(input_tensor)

        # Issue replay without waiting for completion
        if run_mode == "NON_TRACED":
            output = run_nontraced(ttnn_linear, input_tensor)
        else:
            output = ttnn_linear(input_tensor)
        outputs.append(output)

    print(f"  All {num_replays} replays issued")

    # Phase 3: Sync and verify all outputs
    print(f"\n[Rapid Fire Test - {run_mode}] Phase 3: Verifying outputs...")

    # Convert outputs to torch for verification
    verification_errors = 0
    for i, (output, input_tensor) in enumerate(zip(outputs, inputs)):
        if output is None:
            print(f"  ERROR: Output {i} is None")
            verification_errors += 1
            continue

        # Check output is valid tensor
        try:
            if hasattr(output, "to_torch"):
                output_torch = output.to_torch
            elif hasattr(output, "cpu"):
                output_torch = output.cpu()
            else:
                output_torch = output

            # Basic shape check
            expected_shape = list(input_shape[:-1]) + [torch_linear.out_features]
            actual_shape = list(output_torch.shape)

            if actual_shape != expected_shape:
                print(f"  ERROR: Output {i} shape mismatch: expected {expected_shape}, got {actual_shape}")
                verification_errors += 1
        except Exception as e:
            print(f"  ERROR: Output {i} verification failed: {e}")
            verification_errors += 1

    assert verification_errors == 0, f"Found {verification_errors} verification errors"
    print(f"\n[Rapid Fire Test - {run_mode}] SUCCESS: All {num_replays} outputs verified!")
    TracedRun.release_all()


@pytest.mark.timeout(120)
@pytest.mark.skipif(not _is_t3k(), reason="Requires MESH_DEVICE=T3K")
@pytest.mark.parametrize("device_params", [T3K_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize("run_mode", ["NON_TRACED", "TRACED"])
def test_semaphore_numerical_correctness_drift(mesh_device, run_mode):
    """
    Test designed to EXPOSE numerical correctness drift over many replays.

    Hypothesis: Semaphore issues in reduce_scatter can cause partial results
    to be computed incorrectly, leading to numerical drift that grows with
    the number of replays.

    Expected failure: Numerical error exceeds tolerance after N replays,
    indicating the reduce_scatter is producing wrong partial sums.
    """
    TracedRun.release_all()

    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    if num_devices < 2:
        pytest.skip(f"Need at least 2 devices for sharded test, got {num_devices}")

    # Helper function for non-traced execution
    @disable_trace
    def run_nontraced(layer, input_tensor):
        return layer(input_tensor)

    ttnn_linear, torch_linear, in_features = create_sharded_linear(mesh_device, "numerical")
    input_shape = (1, 32, in_features)

    # Use deterministic input for reproducibility
    torch.manual_seed(42)
    deterministic_input = torch.randn(*input_shape, dtype=torch.bfloat16)

    # Compute reference with PyTorch
    with torch.no_grad():
        reference_output = torch_linear(deterministic_input)

    print(f"\n[Numerical Test - {run_mode}] Reference computed with PyTorch")
    print(f"  Reference output shape: {reference_output.shape}")
    print(f"  Reference output range: [{reference_output.min():.4f}, {reference_output.max():.4f}]")

    # Phase 1: Capture trace
    print(f"\n[Numerical Test - {run_mode}] Phase 1: Capturing trace...")
    _ = ttnn_linear(deterministic_input.clone())
    print(f"  Trace capture complete")

    # Phase 2: Run 50 replays with same input, check numerical drift
    num_replays = 50
    tolerance = 0.1  # bfloat16 has limited precision

    print(f"\n[Numerical Test - {run_mode}] Phase 2: Checking numerical drift over {num_replays} replays...")

    errors = []
    max_errors = []

    for i in range(num_replays):
        if run_mode == "NON_TRACED":
            output = run_nontraced(ttnn_linear, deterministic_input.clone())
        else:
            output = ttnn_linear(deterministic_input.clone())

        # Convert to torch for comparison
        try:
            if hasattr(output, "to_torch"):
                output_torch = output.to_torch
            elif hasattr(output, "cpu"):
                output_torch = output.cpu()
            else:
                output_torch = output

            # Compute error vs reference
            diff = (output_torch.float() - reference_output.float()).abs()
            mean_error = diff.mean().item()
            max_error = diff.max().item()

            errors.append(mean_error)
            max_errors.append(max_error)

            if (i + 1) % 10 == 0:
                print(f"  Replay {i+1}: mean_error={mean_error:.6f}, max_error={max_error:.6f}")

            # Check if error is growing (sign of drift)
            if max_error > tolerance:
                print(f"  WARNING: Replay {i+1} max_error {max_error:.6f} exceeds tolerance {tolerance}")

        except Exception as e:
            print(f"  ERROR: Replay {i} comparison failed: {e}")
            errors.append(float("inf"))
            max_errors.append(float("inf"))

    # Analyze drift pattern
    print(f"\n[Numerical Test - {run_mode}] Drift analysis:")
    print(f"  Initial mean error: {errors[0]:.6f}")
    print(f"  Final mean error: {errors[-1]:.6f}")
    print(f"  Max mean error: {max(errors):.6f}")
    print(f"  Max absolute error: {max(max_errors):.6f}")

    # Check for drift: is the final error significantly larger than initial?
    drift_ratio = errors[-1] / (errors[0] + 1e-10)
    print(f"  Drift ratio (final/initial): {drift_ratio:.2f}x")

    # Fail if we see significant numerical drift
    assert max(max_errors) <= tolerance, (
        f"Numerical drift detected: max error {max(max_errors):.6f} > tolerance {tolerance}. "
        f"This may indicate reduce_scatter semaphore issues causing incorrect partial sums."
    )

    # Also fail if error grows significantly over time
    assert drift_ratio < 2.0, (
        f"Error drift detected: final error is {drift_ratio:.2f}x larger than initial. "
        f"This suggests accumulating numerical errors from semaphore issues."
    )

    print(f"\n[Numerical Test - {run_mode}] SUCCESS: Numerical accuracy maintained over {num_replays} replays!")
    TracedRun.release_all()


@pytest.mark.timeout(120)
@pytest.mark.skipif(not _is_t3k(), reason="Requires MESH_DEVICE=T3K")
@pytest.mark.parametrize("device_params", [T3K_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize("run_mode", ["NON_TRACED", "TRACED"])
def test_multi_shape_trace_interleaving(mesh_device, run_mode):
    """
    Test designed to EXPOSE semaphore mismatch across different trace IDs.

    Hypothesis: Running the same layer with different input shapes creates
    multiple traces (each with its own trace ID). Interleaving replays between
    these traces can expose semaphore mismatches because each trace was captured
    with different semaphore state assumptions.

    Expected failure: Hang or incorrect results when switching between traces
    with different shapes.
    """
    TracedRun.release_all()

    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    if num_devices < 2:
        pytest.skip(f"Need at least 2 devices for sharded test, got {num_devices}")

    # Helper function for non-traced execution
    @disable_trace
    def run_nontraced(layer, input_tensor):
        return layer(input_tensor)

    ttnn_linear, _, in_features = create_sharded_linear(mesh_device, "multi_shape")

    # Define 3 different input shapes (varying sequence length)
    shapes = [
        (1, 16, in_features),  # Shape A
        (1, 32, in_features),  # Shape B
        (1, 64, in_features),  # Shape C
    ]

    # Phase 1: Capture traces for all 3 shapes
    print(f"\n[Multi-Shape Test - {run_mode}] Phase 1: Capturing traces for 3 different shapes...")
    for i, shape in enumerate(shapes):
        input_tensor = torch.randn(*shape, dtype=torch.bfloat16)
        _ = ttnn_linear(input_tensor)
        print(f"  Shape {chr(65+i)} {shape} trace captured")

    print(f"  Total trace cache size: {TracedRun.cache_size()}")

    # Phase 2: Interleave 30 rounds of replays in A, B, C pattern
    num_rounds = 30
    print(f"\n[Multi-Shape Test - {run_mode}] Phase 2: Interleaving {num_rounds} rounds (A, B, C pattern)...")

    for round_idx in range(num_rounds):
        if (round_idx + 1) % 10 == 0:
            print(f"  Round {round_idx+1}/{num_rounds}...")

        # Cycle through shapes A, B, C
        for shape_idx, shape in enumerate(shapes):
            input_tensor = torch.randn(*shape, dtype=torch.bfloat16)

            # This may hang if semaphore state mismatches between trace IDs
            if run_mode == "NON_TRACED":
                output = run_nontraced(ttnn_linear, input_tensor)
            else:
                output = ttnn_linear(input_tensor)

            assert output is not None, f"Round {round_idx+1}, Shape {chr(65+shape_idx)} returned None"

    print(f"\n[Multi-Shape Test - {run_mode}] SUCCESS: All {num_rounds} interleaved rounds completed!")
    TracedRun.release_all()


@pytest.mark.timeout(120)
@pytest.mark.skipif(not _is_t3k(), reason="Requires MESH_DEVICE=T3K")
@pytest.mark.parametrize("device_params", [T3K_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize("run_mode", ["NON_TRACED", "TRACED"])
def test_alternating_traced_nontraced_execution(mesh_device, run_mode):
    """
    Test designed to EXPOSE semaphore state corruption at mode boundaries.

    Hypothesis: Alternating between traced replay and non-traced execution
    can corrupt semaphore state because non-traced execution may modify
    semaphore values that traced replay expects to be in a specific state.

    Expected failure: Hang or incorrect results when switching between
    traced and non-traced execution modes.
    """
    if run_mode == "NON_TRACED":
        pytest.skip("This test only makes sense in TRACED mode")

    TracedRun.release_all()

    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    if num_devices < 2:
        pytest.skip(f"Need at least 2 devices for sharded test, got {num_devices}")

    ttnn_linear, _, in_features = create_sharded_linear(mesh_device, "alternating")
    input_shape = (1, 32, in_features)

    # Phase 1: Capture trace with normal execution
    print(f"\n[Alternating Test - {run_mode}] Phase 1: Capturing trace...")
    input1 = torch.randn(*input_shape, dtype=torch.bfloat16)
    _ = ttnn_linear(input1)
    print(f"  Trace capture complete. Cache size: {TracedRun.cache_size()}")

    # Define a non-traced execution wrapper using disable_trace decorator
    @disable_trace
    def run_nontraced(layer, input_tensor):
        return layer(input_tensor)

    # Phase 2: Alternate between traced and non-traced execution
    num_cycles = 20
    print(f"\n[Alternating Test - {run_mode}] Phase 2: Alternating {num_cycles} cycles (traced/non-traced)...")

    for cycle in range(num_cycles):
        if (cycle + 1) % 5 == 0:
            print(f"  Cycle {cycle+1}/{num_cycles}...")

        # Step 1: Traced replay
        input_traced = torch.randn(*input_shape, dtype=torch.bfloat16)
        output_traced = ttnn_linear(input_traced)
        assert output_traced is not None, f"Cycle {cycle+1} traced execution returned None"

        # Step 2: Non-traced execution (bypasses trace replay)
        input_nontraced = torch.randn(*input_shape, dtype=torch.bfloat16)
        output_nontraced = run_nontraced(ttnn_linear, input_nontraced)
        assert output_nontraced is not None, f"Cycle {cycle+1} non-traced execution returned None"

    print(f"\n[Alternating Test - {run_mode}] SUCCESS: All {num_cycles} alternating cycles completed!")
    TracedRun.release_all()


@pytest.mark.timeout(180)
@pytest.mark.skipif(not _is_t3k(), reason="Requires MESH_DEVICE=T3K")
@pytest.mark.parametrize("device_params", [T3K_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize("run_mode", ["NON_TRACED", "TRACED"])
def test_chained_ccl_pipeline(mesh_device, run_mode):
    """
    Test designed to EXPOSE semaphore exhaustion faster through pipeline execution.

    Hypothesis: Running input through multiple sharded linear layers in sequence
    (a pipeline) consumes 3x semaphores per pass compared to single-layer execution.
    This accelerates semaphore exhaustion and should trigger failures sooner.

    Expected failure: Hang/timeout or incorrect results due to faster semaphore
    depletion from pipeline execution.
    """
    TracedRun.release_all()

    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    if num_devices < 2:
        pytest.skip(f"Need at least 2 devices for sharded test, got {num_devices}")

    # Helper function for non-traced execution
    @disable_trace
    def run_nontraced(layer, input_tensor):
        return layer(input_tensor)

    # Create 3 sharded linear layers as a pipeline with matching dimensions
    # layer1: in=2048 (256*8), out=512
    # layer2: in=512, out=256
    # layer3: in=256, out=128
    # This ensures each layer's output matches the next layer's input
    print(f"\n[Pipeline Test - {run_mode}] Creating 3-layer pipeline...")
    layers = []

    # Define pipeline dimensions: each layer's output matches next layer's input
    pipeline_dims = [
        (256 * num_devices, 512),  # layer0: in=2048, out=512
        (512, 256),  # layer1: in=512, out=256
        (256, 128),  # layer2: in=256, out=128
    ]

    for i, (in_feat, out_feat) in enumerate(pipeline_dims):
        ttnn_linear, _, _ = create_sharded_linear(
            mesh_device, f"pipeline_{i}", in_features=in_feat, out_features=out_feat
        )
        layers.append(ttnn_linear)
        print(f"  Layer {i} created (in={in_feat}, out={out_feat})")

    input_shape = (1, 32, pipeline_dims[0][0])

    # Phase 1: Capture traces for all layers in pipeline
    print(f"\n[Pipeline Test - {run_mode}] Phase 1: Capturing traces for pipeline...")

    # Run through the pipeline once to capture all traces
    pipeline_input = torch.randn(*input_shape, dtype=torch.bfloat16)
    x = pipeline_input
    for i, layer in enumerate(layers):
        x = layer(x)
        print(f"  Layer {i} trace captured")

    print(f"  Total trace cache size: {TracedRun.cache_size()}")

    # Phase 2: Run 50 pipeline replays
    num_replays = 50
    print(f"\n[Pipeline Test - {run_mode}] Phase 2: Running {num_replays} pipeline replays...")

    for replay in range(num_replays):
        if (replay + 1) % 10 == 0:
            print(f"  Pipeline replay {replay+1}/{num_replays}...")

        # Run through entire pipeline
        x = torch.randn(*input_shape, dtype=torch.bfloat16)
        for layer_idx, layer in enumerate(layers):
            if run_mode == "NON_TRACED":
                x = run_nontraced(layer, x)
            else:
                x = layer(x)
            assert x is not None, f"Replay {replay+1}, Layer {layer_idx} returned None"

    print(f"\n[Pipeline Test - {run_mode}] SUCCESS: All {num_replays} pipeline replays completed!")
    TracedRun.release_all()


@pytest.mark.timeout(120)
@pytest.mark.skipif(not _is_t3k(), reason="Requires MESH_DEVICE=T3K")
@pytest.mark.parametrize("device_params", [T3K_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize("run_mode", ["NON_TRACED", "TRACED"])
def test_cross_output_corruption(mesh_device, run_mode):
    """
    Test designed to EXPOSE buffer aliasing if outputs share memory.

    Hypothesis: In traced mode, output buffers may be reused/aliased, causing
    outputs from different replays to reference the same memory. This test runs
    multiple replays with deterministic inputs, stores ALL outputs, and verifies
    each output AFTER all replays complete.

    Expected failure: All stored outputs contain the same values (aliased to
    the last replay's output buffer) instead of distinct values.
    """
    TracedRun.release_all()

    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    if num_devices < 2:
        pytest.skip(f"Need at least 2 devices for sharded test, got {num_devices}")

    ttnn_linear, torch_linear, in_features = create_sharded_linear(mesh_device, "cross_output")
    input_shape = (1, 32, in_features)

    # Helper function for non-traced execution
    @disable_trace
    def run_nontraced(layer, input_tensor):
        return layer(input_tensor)

    # Phase 1: Capture trace
    print(f"\n[Cross-Output Test - {run_mode}] Phase 1: Capturing trace...")
    input1 = torch.randn(*input_shape, dtype=torch.bfloat16)
    _ = ttnn_linear(input1)
    print(f"  Trace capture complete")

    # Phase 2: Run 20 replays with deterministic inputs, store ALL outputs
    num_replays = 20
    print(f"\n[Cross-Output Test - {run_mode}] Phase 2: Running {num_replays} replays with deterministic inputs...")

    stored_outputs = []
    stored_inputs = []
    expected_outputs = []

    for i in range(num_replays):
        if (i + 1) % 5 == 0:
            print(f"  Replay {i+1}/{num_replays}...")

        # Use deterministic input based on index
        torch.manual_seed(1000 + i)
        input_tensor = torch.randn(*input_shape, dtype=torch.bfloat16)
        stored_inputs.append(input_tensor.clone())

        # Compute expected output with PyTorch
        with torch.no_grad():
            expected = torch_linear(input_tensor)
            expected_outputs.append(expected.clone())

        # Run through TTNN layer
        if run_mode == "NON_TRACED":
            output = run_nontraced(ttnn_linear, input_tensor)
        else:
            output = ttnn_linear(input_tensor)
        stored_outputs.append(output)

    # Phase 3: AFTER all replays complete, verify each stored output
    print(f"\n[Cross-Output Test - {run_mode}] Phase 3: Verifying {num_replays} stored outputs...")

    aliasing_detected = False
    verification_errors = 0

    # Convert all outputs to torch tensors first
    torch_outputs = []
    for i, output in enumerate(stored_outputs):
        try:
            if hasattr(output, "to_torch"):
                torch_outputs.append(output.to_torch.clone())
            elif hasattr(output, "cpu"):
                torch_outputs.append(output.cpu().clone())
            else:
                torch_outputs.append(output.clone())
        except Exception as e:
            print(f"  ERROR: Failed to convert output {i}: {e}")
            torch_outputs.append(None)
            verification_errors += 1

    # Check for aliasing: are all outputs identical?
    if len(torch_outputs) > 1 and torch_outputs[0] is not None:
        identical_count = 0
        for i in range(1, len(torch_outputs)):
            if torch_outputs[i] is not None:
                if torch.allclose(torch_outputs[0], torch_outputs[i], atol=1e-6):
                    identical_count += 1

        if identical_count == len(torch_outputs) - 1:
            print(f"  WARNING: All {num_replays} outputs are IDENTICAL - possible buffer aliasing!")
            aliasing_detected = True

    # Verify each output matches expected (within tolerance)
    tolerance = 0.1  # bfloat16 tolerance
    for i, (torch_output, expected) in enumerate(zip(torch_outputs, expected_outputs)):
        if torch_output is None:
            continue

        diff = (torch_output.float() - expected.float()).abs()
        max_error = diff.max().item()

        if max_error > tolerance:
            print(f"  Output {i}: max_error={max_error:.6f} (exceeds tolerance {tolerance})")
            verification_errors += 1
        elif (i + 1) % 5 == 0:
            print(f"  Output {i}: max_error={max_error:.6f} OK")

    assert not aliasing_detected, (
        "Buffer aliasing detected: all outputs are identical despite different inputs. "
        "This indicates outputs are sharing the same memory buffer."
    )

    assert verification_errors == 0, f"Found {verification_errors} verification errors"

    print(f"\n[Cross-Output Test - {run_mode}] SUCCESS: All {num_replays} outputs verified as distinct!")
    TracedRun.release_all()


@pytest.mark.timeout(120)
@pytest.mark.skipif(not _is_t3k(), reason="Requires MESH_DEVICE=T3K")
@pytest.mark.parametrize("device_params", [T3K_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [MESH_DEVICE_MAP.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))],
    indirect=True,
)
@pytest.mark.parametrize("run_mode", ["NON_TRACED", "TRACED"])
def test_dual_trace_same_pattern_replay(mesh_device, run_mode):
    """
    Test designed to EXPOSE semaphore leakage between trace contexts.

    Hypothesis: Creating multiple traces with different shapes but replaying
    them with the same random seed pattern can expose semaphore leakage if
    semaphore state from one trace context affects another.

    Expected failure: Outputs from same-seed replays are inconsistent,
    indicating semaphore state contamination between trace contexts.
    """
    TracedRun.release_all()

    num_devices = mesh_device.get_num_devices() if hasattr(mesh_device, "get_num_devices") else 1
    if num_devices < 2:
        pytest.skip(f"Need at least 2 devices for sharded test, got {num_devices}")

    # Helper function for non-traced execution
    @disable_trace
    def run_nontraced(layer, input_tensor):
        return layer(input_tensor)

    ttnn_linear, _, in_features = create_sharded_linear(mesh_device, "dual_trace")

    # Define 2 different shapes
    shape_a = (1, 16, in_features)
    shape_b = (1, 32, in_features)

    # Phase 1: Capture traces for both shapes
    print(f"\n[Dual-Trace Test - {run_mode}] Phase 1: Capturing traces for 2 different shapes...")

    input_a = torch.randn(*shape_a, dtype=torch.bfloat16)
    _ = ttnn_linear(input_a)
    print(f"  Shape A {shape_a} trace captured")

    input_b = torch.randn(*shape_b, dtype=torch.bfloat16)
    _ = ttnn_linear(input_b)
    print(f"  Shape B {shape_b} trace captured")

    print(f"  Total trace cache size: {TracedRun.cache_size()}")

    # Phase 2: Interleave replays with same random seed for consistency check
    num_rounds = 20
    seed_base = 5000
    print(f"\n[Dual-Trace Test - {run_mode}] Phase 2: Running {num_rounds} interleaved rounds...")

    # Store outputs for consistency verification
    outputs_a = []
    outputs_b = []
    reference_outputs_a = {}
    reference_outputs_b = {}

    for round_idx in range(num_rounds):
        if (round_idx + 1) % 5 == 0:
            print(f"  Round {round_idx+1}/{num_rounds}...")

        # Run shape A with seed
        seed = seed_base + round_idx
        torch.manual_seed(seed)
        input_a = torch.randn(*shape_a, dtype=torch.bfloat16)
        if run_mode == "NON_TRACED":
            output_a = run_nontraced(ttnn_linear, input_a)
        else:
            output_a = ttnn_linear(input_a)
        assert output_a is not None, f"Round {round_idx+1} Shape A returned None"

        # Store or verify output
        try:
            if hasattr(output_a, "to_torch"):
                output_a_torch = output_a.to_torch.clone()
            elif hasattr(output_a, "cpu"):
                output_a_torch = output_a.cpu().clone()
            else:
                output_a_torch = output_a.clone()
            outputs_a.append((seed, output_a_torch))
        except Exception:
            outputs_a.append((seed, None))

        # Run shape B with different seed
        seed_b = seed_base + num_rounds + round_idx
        torch.manual_seed(seed_b)
        input_b = torch.randn(*shape_b, dtype=torch.bfloat16)
        if run_mode == "NON_TRACED":
            output_b = run_nontraced(ttnn_linear, input_b)
        else:
            output_b = ttnn_linear(input_b)
        assert output_b is not None, f"Round {round_idx+1} Shape B returned None"

        # Store or verify output
        try:
            if hasattr(output_b, "to_torch"):
                output_b_torch = output_b.to_torch.clone()
            elif hasattr(output_b, "cpu"):
                output_b_torch = output_b.cpu().clone()
            else:
                output_b_torch = output_b.clone()
            outputs_b.append((seed_b, output_b_torch))
        except Exception:
            outputs_b.append((seed_b, None))

    # Phase 3: Replay with same seeds and verify consistency
    print(f"\n[Dual-Trace Test - {run_mode}] Phase 3: Verifying consistency with same seeds...")

    consistency_errors = 0

    # Re-run a subset with same seeds and compare
    for i in range(min(5, num_rounds)):
        seed, original_output = outputs_a[i]

        torch.manual_seed(seed)
        input_a = torch.randn(*shape_a, dtype=torch.bfloat16)
        if run_mode == "NON_TRACED":
            replay_output = run_nontraced(ttnn_linear, input_a)
        else:
            replay_output = ttnn_linear(input_a)

        try:
            if hasattr(replay_output, "to_torch"):
                replay_torch = replay_output.to_torch
            elif hasattr(replay_output, "cpu"):
                replay_torch = replay_output.cpu()
            else:
                replay_torch = replay_output

            if original_output is not None:
                diff = (replay_torch.float() - original_output.float()).abs()
                max_diff = diff.max().item()

                if max_diff > 0.01:  # Should be exactly the same
                    print(f"  Inconsistency detected for seed {seed}: max_diff={max_diff:.6f}")
                    consistency_errors += 1
                else:
                    print(f"  Seed {seed}: consistent (max_diff={max_diff:.6f})")

        except Exception as e:
            print(f"  ERROR comparing output for seed {seed}: {e}")
            consistency_errors += 1

    assert consistency_errors == 0, (
        f"Found {consistency_errors} consistency errors. "
        "Same seed should produce identical outputs, but they differ. "
        "This indicates semaphore leakage between trace contexts."
    )

    print(f"\n[Dual-Trace Test - {run_mode}] SUCCESS: All same-seed outputs are consistent!")
    TracedRun.release_all()
