import torch, ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

device = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    x = ttnn.from_torch(
        torch.randn(1, 1, 32, 320, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16
    )

    # Replicate what the cpp engine does: run the body (incl. the readback) to
    # completion INSIDE one NO_DISPATCH collect window.
    ttnn.graph.up_front_begin_collect()
    try:
        out = groupnorm_sc_N_1_HW_C(x, 10)  # the op (generic_op) -> should be collected
        t = ttnn.to_torch(out)  # THE RISK: readback on addr-0 under NO_DISPATCH
        print("to_torch under NO_DISPATCH returned shape", tuple(t.shape))
    except Exception as e:
        print("body raised under NO_DISPATCH (expected, swallowed):", repr(e)[:160])
    finally:
        ttnn.graph.up_front_end_collect()

    print("collected", ttnn.graph.up_front_num_collected(), "/ unique", ttnn.graph.up_front_num_unique())
    n_prog, n_err, used, wall = ttnn.graph.up_front_compile(device, 4, True)
    print(f"compiled {n_prog} programs, errors={n_err}, workers={used}, wall={wall:.2f}s")

    out2 = groupnorm_sc_N_1_HW_C(x, 10)  # warm run must still be correct
    g = ttnn.to_torch(out2).float()
    print("warm out shape", tuple(g.shape), "mean", round(g.mean().item(), 4), "std", round(g.std().item(), 4))
finally:
    ttnn.close_device(device)
