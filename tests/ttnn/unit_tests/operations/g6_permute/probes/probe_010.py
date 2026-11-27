import torch, ttnn
from ttnn.operations.g6_permute import g6_permute, g6_permute_plan
from eval.golden_tests.g6_permute.helpers import check_bf8_one_step, _bf8_host_rendering

device = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    for shape, mem in [((1, 128, 256, 32), ttnn.L1_MEMORY_CONFIG), ((1, 64, 64, 64), ttnn.DRAM_MEMORY_CONFIG)]:
        x = torch.randn(shape).to(torch.bfloat16)
        t = ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
        plan = g6_permute_plan(t, (0, 3, 1, 2))
        print(
            "plan",
            plan.regime,
            plan.num_cores,
            plan.groups_per_core_g1,
            plan.group_depth,
            plan.l1_bytes_per_core,
            plan.out_l1_bytes_per_core,
        )
        out = g6_permute(t, (0, 3, 1, 2))
        logical = ttnn.to_torch(t).to(torch.float32)
        permuted = torch.permute(logical, (0, 3, 1, 2)).contiguous()
        expected = _bf8_host_rendering(permuted)
        actual = ttnn.to_torch(out).to(torch.float32)
        print(shape, out.shape, out.dtype, out.memory_config().buffer_type)
        diff = (actual - expected).abs()
        print("max diff", diff.max().item(), "exact frac", (actual == expected).float().mean().item())
        try:
            check_bf8_one_step(actual, expected, permuted)
            print("ONE-STEP PASS")
        except AssertionError as e:
            print("ONE-STEP FAIL", str(e)[:400])
finally:
    ttnn.close_device(device)
