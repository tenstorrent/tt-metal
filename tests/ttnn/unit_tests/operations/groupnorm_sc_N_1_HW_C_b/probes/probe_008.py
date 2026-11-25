import ttnn, dataclasses
import ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b_program_descriptor as pd

dev = ttnn.open_device(device_id=0)
try:
    new_fn = pd.l1_cb_budget
    old_budget = ttnn.get_max_worker_l1_unreserved_size()
    for shape, G in [((1, 1, 65536, 512), 32), ((1, 1, 16384, 1152), 32), ((1, 1, 16384, 256), 16)]:
        t = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
        )
        pd.l1_cb_budget = new_fn
        a = pd.plan_or_raise(t, G, has_gamma=True, has_beta=True)
        pd.l1_cb_budget = lambda d: old_budget
        b = pd.plan_or_raise(t, G, has_gamma=True, has_beta=True)
        diff = {
            f.name: (repr(getattr(a, f.name)), repr(getattr(b, f.name)))
            for f in dataclasses.fields(a)
            if repr(getattr(a, f.name)) != repr(getattr(b, f.name))
        }
        print("PLANCMP", shape, G, "budgets", new_fn(dev), old_budget, "differs only in:", diff)
        t.deallocate()
finally:
    pd.l1_cb_budget = new_fn
    ttnn.close_device(dev)
