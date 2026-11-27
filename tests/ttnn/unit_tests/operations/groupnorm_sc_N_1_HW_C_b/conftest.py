import pytest

from ttnn.operations.groupnorm_sc_N_1_HW_C_b import validate


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.use_module_device)


def plan_regime(plan):
    """Regime names for a planner result: residency (full / partial / stream) plus "multi" when a
    core runs several statistic units back to back (cpi = 1 and U > P)."""
    k, b = plan["K_alloc"], plan["B_max"]
    regime = {"full" if k >= b else ("stream" if k == 0 else "partial")}
    if plan["cpi"] == 1 and plan["U"] > plan["P"]:
        regime.add("multi")
    return regime


@pytest.fixture
def assert_regime():
    """Checks that the default planner puts these tensors in the regime a test is meant to cover.
    The regime shapes are sized for a 110-core Blackhole grid; on another grid the check fails
    instead of the test silently covering a different path."""

    def check(want, tt_input, num_groups, *, gamma=None, beta=None, eps=1e-5, in_place=False):
        plan = validate(tt_input, num_groups, gamma=gamma, beta=beta, eps=eps, in_place=in_place)
        got = plan_regime(plan)
        assert got == set(want), f"planner regime {sorted(got)} != {sorted(want)}: {plan}"

    return check


@pytest.fixture
def assert_inputs_regime(device, assert_regime):
    """assert_regime for an `inputs` tuple as run_groupnorm_sc_N_1_HW_C takes it. Allocates
    uninitialized tensors (no host transfer) just for the plan, then frees them."""
    import ttnn
    from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import block_shard_memory_config, placement_of

    def check(want, inputs, *, dtype, layout, affine_dtype, affine_layout, in_place=False):
        shape, num_groups, grid = inputs[0], inputs[1], placement_of(inputs)
        mem = (
            ttnn.DRAM_MEMORY_CONFIG
            if grid is None
            else block_shard_memory_config(shape, grid, layout=layout, dtype=dtype, device=device)
        )
        x = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, layout, device, mem)
        g = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 1, shape[-1]]), affine_dtype, affine_layout, device, ttnn.DRAM_MEMORY_CONFIG
        )
        try:
            assert_regime(want, x, num_groups, gamma=g, beta=g, in_place=in_place)
        finally:
            x.deallocate()
            g.deallocate()

    return check
