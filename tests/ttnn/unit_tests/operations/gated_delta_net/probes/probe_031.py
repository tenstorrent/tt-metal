import torch, ttnn, importlib
from ttnn.operations.gated_delta_net import gated_delta_net
from tests.ttnn.unit_tests.operations.gated_delta_net.test_gated_delta_net import ORDER, compute_config, make_inputs

pd = importlib.import_module("ttnn.operations.gated_delta_net.gated_delta_net_program_descriptor")
ref = torch.load("/tmp/gdn13/ref.pt")
device = ttnn.open_device(device_id=0)
cases = {
    "decode_b1_bf16": (((1, 1, 2560), (1, 12, 128, 128)), ttnn.bfloat16),
    "decode_b16_bf16": (((1, 16, 2560), (16, 12, 128, 128)), ttnn.bfloat16),
    "decode_b1_f32": (((1, 1, 2560), (1, 12, 128, 128)), ttnn.float32),
    "prefill_t128": (((1, 128, 2560), (1, 12, 128, 128)), ttnn.bfloat16),
    "prefill_t96": (((1, 96, 2560), (1, 12, 128, 128)), ttnn.bfloat16),
    "batched_b2_t40": (((1, 80, 2560), (2, 12, 128, 128)), ttnn.bfloat16),
    "small_f32": (((1, 64, 384), (1, 4, 32, 32)), ttnn.float32),
}
bad = 0
for rnd in range(2):  # round 1 hits the cache with fresh tensors
    for name, (inputs, dt) in cases.items():
        torch.manual_seed(0)
        tensors = make_inputs(inputs)
        for in_place in (False, True):
            junk = ttnn.from_torch(torch.zeros(1, 32, 32), device=device)  # shifts allocations between rounds
            dev = [
                ttnn.from_torch(
                    tensors[n],
                    dtype=ttnn.float32 if n == "recurrent_state" else dt,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for n in ORDER
            ]
            kw = dict(conv_state_out=dev[4], recurrent_state_out=dev[5]) if in_place else {}
            outs = gated_delta_net(*dev, compute_kernel_config=compute_config(), **kw)
            got = [ttnn.to_torch(t) for t in outs]
            ok = all(torch.equal(g, r) for g, r in zip(got, ref[(name, in_place)]))
            bad += not ok
            print(f"CHECK round{rnd} {name} in_place={in_place}: {'BITEXACT' if ok else 'DIFF'}")
print("CACHE entries", pd.program_cache_size(), "bad", bad)
ttnn.close_device(device)
