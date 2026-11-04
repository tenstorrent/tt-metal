import time, cProfile, pstats, io, os, sys
import torch, ttnn

sys.path.insert(0, os.getcwd())
from eval.golden_tests.nomic_moe_combine.helpers import (
    route,
    make_expert_rows,
    make_bias,
    rows_tensor,
    index_tensor,
    weights_tensor,
    bias_tensor,
    output_config,
)
from ttnn.operations.nomic_moe_combine import nomic_moe_combine, prepare_combine_bias

device = ttnn.open_device(device_id=0, l1_small_size=0)
try:
    for batch in (8, 4, 2):
        seqlen = 512
        tokens = batch * seqlen
        shape = (batch, 1, seqlen, 768)
        config = output_config("norm_layout", shape, device)
        plan = route(tokens, "real_text", seed=1)
        values = make_expert_rows(plan["rows"], plan["slot_row"], seed=1)
        rows = rows_tensor(values, device)
        idx = index_tensor(plan["combine_index"], device)
        w = weights_tensor(plan["combine_weights"], device)
        prepared = prepare_combine_bias(bias_tensor(make_bias(1), device), device)
        call = lambda: nomic_moe_combine(rows, idx, w, prepared, num_tokens=tokens, batch=batch, memory_config=config)
        outs = [call() for _ in range(5)]
        ttnn.synchronize_device(device)
        for o in outs:
            ttnn.deallocate(o)
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            o = call()
            times.append(time.perf_counter() - t0)
            ttnn.synchronize_device(device)
            ttnn.deallocate(o)
        times.sort()
        print(f"HOST {batch}x{seqlen}: median {1e6*times[25]:.1f} us, min {1e6*times[0]:.1f} us")
        if batch == 8 and os.environ.get("PROF"):
            pr = cProfile.Profile()
            pr.enable()
            for _ in range(50):
                o = call()
                ttnn.synchronize_device(device)
                ttnn.deallocate(o)
            pr.disable()
            s = io.StringIO()
            pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(25)
            print(s.getvalue())
finally:
    ttnn.close_device(device)
