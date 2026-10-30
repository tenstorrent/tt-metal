import sys, torch, ttnn

sys.path.insert(0, "/tmp/claude-1211408582/-localdev-iwrosz/2424f38f-ecba-4b65-bab4-92f5bc368962/scratchpad/spikes")
import spike_b_inplace_pool as sb
from tests.ttnn.unit_tests.operations.transformers.gdn_decode_test_utils import make_inputs

device = ttnn.open_device(device_id=0)
try:
    mc = sb.pool_config(device)
    print("grid", device.compute_with_storage_grid_size(), "cores", sb.head_cores(device), flush=True)
    x = make_inputs(sb.B, 1, sb.H, sb.HV, sb.K, sb.V, seed=101)
    s0 = x["initial_state"]
    print("fresh...", flush=True)
    o_ref, s_ref = sb.fresh_step(device, x, s0)
    print("fresh done", flush=True)
    pool = sb.dev(device, s0.reshape(sb.BH, sb.K, sb.V), mc)
    print("pool at", hex(pool.buffer_address()), pool.memory_config(), flush=True)
    print("inplace...", flush=True)
    o, s = sb.inplace_step(device, x, pool)
    print(
        "inplace done; state equal",
        torch.equal(ttnn.to_torch(s).reshape(sb.B, sb.HV, sb.K, sb.V), s_ref),
        "o equal",
        torch.equal(o, o_ref),
        flush=True,
    )
finally:
    ttnn.close_device(device)
