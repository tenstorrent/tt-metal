"""
TTTv2 Developer Experience Example

This demonstrates the two-step development flow:
1. Get it working (architecture-first)
2. Make it fast (performance optimization)
"""

# =============================================================================
# STEP 1: Architecture-First Development (Get It Working)
# =============================================================================

print("=== Step 1: Get It Working ===\n")

# Option A: From HuggingFace
from tt_transformers_v2 import TTTModel

model = TTTModel.from_huggingface("meta-llama/Llama-3-8b", device="ttnn:0")
print("✓ Model loaded from HuggingFace")

# Option B: From custom architecture
from tt_transformers_v2.specs import ArchitectureSpec

# Define pure architecture - no hardware details!
arch = ArchitectureSpec(
    model_type="decoder",
    vocab_size=32000,
    max_sequence_length=8192,
    layers=[
        # ... layer specs
    ],
)
model = TTTModel.from_spec(arch, device="ttnn:0")
print("✓ Model created from custom spec")

# Just works with default hardware configs!
output = model.generate("The meaning of life is", max_tokens=50)
print(f"✓ Generated: {output}")
print("\nNote: Using default hardware configs - might not be optimal")

# Check compatibility
issues = model.check_hardware_compatibility()
if issues:
    print("\n⚠️  Compatibility issues detected:")
    for issue in issues:
        print(f"  - {issue['module']}: {issue['issue']}")
        print(f"    Suggestion: {issue['suggestion']}")
else:
    print("✓ No compatibility issues")

# =============================================================================
# STEP 2: Performance Optimization (Make It Fast)
# =============================================================================

print("\n\n=== Step 2: Make It Fast ===\n")

# Profile to find bottlenecks
print("Running performance profiling...")
sample_input = model.tokenizer("A long test sequence " * 100)
profile = model.profile(sample_input, num_runs=10)

print("\nBottleneck Analysis:")
print("-" * 60)
print(f"{'Layer':<20} {'Actual':<10} {'Expected':<10} {'Status':<10}")
print("-" * 60)

for bottleneck in profile.bottlenecks():
    status = "🔥 SLOW" if bottleneck["actual_ms"] > bottleneck["expected_ms"] * 1.5 else "✓ OK"
    print(f"{bottleneck['layer']:<20} {bottleneck['actual_ms']:<10.2f} {bottleneck['expected_ms']:<10.2f} {status}")

# Get optimization suggestions
print("\n📊 Performance Insights:")
worst_layer = profile.bottlenecks()[0]
print(f"\nWorst performing layer: {worst_layer['layer']}")
print("Suggestions:")
for i, suggestion in enumerate(worst_layer["suggestions"], 1):
    print(f"  {i}. {suggestion}")

# Apply optimizations interactively
print("\n🔧 Applying Optimizations...")

# Example 1: Quick preset
print("\n1. Trying balanced optimization preset...")
model.apply_optimization_preset("balanced")
new_profile = model.profile(sample_input, num_runs=5)
print(f"   Result: {profile.total_time_ms:.1f}ms → {new_profile.total_time_ms:.1f}ms")
speedup = profile.total_time_ms / new_profile.total_time_ms
print(f"   Speedup: {speedup:.2f}x")

# Example 2: Targeted optimization
print("\n2. Applying targeted optimization to attention layers...")
for i in range(model.num_layers):
    if f"layer_{i}_attention" in [b["layer"] for b in profile.bottlenecks()]:
        # Optimize this specific attention layer
        old_config = model.layers[i].attention.hw_config
        new_config = old_config.copy()
        new_config.device_specific.update(
            {"score_dtype": "HiFi4", "use_flash_attention": True, "kv_cache_mode": "paged"}
        )
        model.layers[i].attention.apply_hw_config(new_config)
        print(f"   ✓ Optimized layer {i} attention")

# Example 3: Memory optimization
print("\n3. Checking memory usage...")
memory_report = model.analyze_memory_usage()
print(f"   Total memory: {memory_report.total_gb:.1f}GB")
print(f"   Activation memory: {memory_report.activation_gb:.1f}GB")
print(f"   Parameter memory: {memory_report.parameter_gb:.1f}GB")

if memory_report.total_gb > 40:  # TTNN chip memory limit
    print("\n   ⚠️  Memory exceeds chip capacity!")
    print("   Applying memory optimizations...")

    # Enable sharding
    model.apply_sharding_strategy("column_parallel", num_chips=8)
    print("   ✓ Enabled column-parallel sharding across 8 chips")

    # Enable activation checkpointing for some layers
    for i in range(0, model.num_layers, 2):  # Every other layer
        model.layers[i].enable_activation_checkpointing()
    print("   ✓ Enabled activation checkpointing for 50% of layers")

    new_memory = model.analyze_memory_usage()
    print(f"   Result: {memory_report.total_gb:.1f}GB → {new_memory.total_gb:.1f}GB")

# =============================================================================
# STEP 2.5: Advanced Debugging
# =============================================================================

print("\n\n=== Advanced Debugging ===\n")

# Isolated module testing
print("Testing individual modules for accuracy...")
test_results = {}

for layer_name, expected_accuracy in [
    ("layer_15_attention", 0.999),
    ("layer_15_ffn", 0.998),
]:
    module = model.get_module(layer_name)
    accuracy = module.test_accuracy(reference_implementation="pytorch")
    test_results[layer_name] = accuracy

    status = "✓" if accuracy >= expected_accuracy else "❌"
    print(f"{status} {layer_name}: {accuracy:.4f} (expected ≥ {expected_accuracy})")

# Debug specific failure
if any(
    acc < expected
    for acc, (_, expected) in zip(
        test_results.values(),
        [
            ("layer_15_attention", 0.999),
            ("layer_15_ffn", 0.998),
        ],
    )
):
    print("\nDebugging accuracy issue in layer_15_attention...")

    # Get detailed error analysis
    module = model.get_module("layer_15_attention")
    error_report = module.analyze_numerical_errors()

    print(f"  Max absolute error: {error_report.max_abs_error:.2e}")
    print(f"  Error location: {error_report.error_location}")
    print(f"  Likely cause: {error_report.likely_cause}")
    print(f"  Suggestion: {error_report.suggestion}")

# =============================================================================
# Final Optimized Configuration
# =============================================================================

print("\n\n=== Final Configuration ===\n")

# Export the optimized configuration
config = model.export_hw_config()
print("Optimized hardware configuration:")
print(config.to_yaml())

# Save for reproducibility
model.save_config("llama3_8b_optimized_ttnn.yaml")
print("\n✓ Configuration saved to llama3_8b_optimized_ttnn.yaml")

# Performance summary
print("\n📈 Performance Summary:")
print(f"  Initial latency: {profile.total_time_ms:.1f}ms")
print(f"  Optimized latency: {new_profile.total_time_ms:.1f}ms")
print(f"  Speedup: {speedup:.2f}x")
print(f"  Memory efficiency: {(1 - new_memory.total_gb/memory_report.total_gb)*100:.1f}%")

# =============================================================================
# Developer Experience Benefits
# =============================================================================

print("\n\n=== Developer Experience Summary ===\n")

print("✅ Benefits of this approach:")
print("  1. Started with working model in < 1 minute")
print("  2. No hardware knowledge needed initially")
print("  3. Clear performance bottlenecks identified")
print("  4. Actionable optimization suggestions")
print("  5. Isolated debugging of specific modules")
print("  6. Reproducible optimized configuration")

print("\n🎯 Key insight: Separate 'what to compute' from 'how to compute it'")
print("   - Step 1: Define architecture (what)")
print("   - Step 2: Optimize execution (how)")

# =============================================================================
# What This Enables
# =============================================================================

print("\n\n=== Enabled Workflows ===\n")

print("1. Rapid Experimentation:")
print("   - Try different architectures without hardware expertise")
print("   - Quick iteration on model design")

print("\n2. Performance Debugging:")
print("   - Profile → Identify bottleneck → Apply fix → Verify")
print("   - Clear feedback loop")

print("\n3. Hardware Portability:")
print("   - Same model code runs on different TT hardware")
print("   - Configs can be shared/versioned")

print("\n4. Team Collaboration:")
print("   - ML researchers focus on architecture")
print("   - Performance engineers optimize configs")
print("   - Clear interface between concerns")

# =============================================================================
# Error Handling Examples
# =============================================================================

print("\n\n=== Common Errors and Solutions ===\n")

# Example error scenarios
try:
    # Scenario 1: Tensor too large
    large_model = TTTModel.from_spec(
        ArchitectureSpec(model_type="decoder", vocab_size=100000, max_sequence_length=100000, layers=[...]),
        device="ttnn:0",
    )
except ShardingError as e:
    print(f"Error handled gracefully:")
    print(f"  Module: {e.module}")
    print(f"  Issue: {e.issue}")
    print(f"  Suggestion: {e.suggestion}")

# More error scenarios with helpful messages...
