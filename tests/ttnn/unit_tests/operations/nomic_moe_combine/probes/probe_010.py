import time, os, sys
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
import importlib

op = importlib.import_module("ttnn.operations.nomic_moe_combine.nomic_moe_combine")
pdm = importlib.import_module("ttnn.operations.nomic_moe_combine.nomic_moe_combine_program_descriptor")


def t(name, fn, n=200, sync=None):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
        if sync:
            sync()
    ts.sort()
    print(f"PIECE {name}: {1e6*ts[n//2]:.2f} us")


device = ttnn.open_device(device_id=0, l1_small_size=0)
try:
    for batch in (8, 2):
        seqlen = 512
        tokens = batch * seqlen
        shape = (batch, 1, seqlen, 768)
        config = output_config("norm_layout", shape, device)
        plan = route(tokens, "real_text", seed=1)
        rows = rows_tensor(make_expert_rows(plan["rows"], plan["slot_row"], seed=1), device)
        idx = index_tensor(plan["combine_index"], device)
        w = weights_tensor(plan["combine_weights"], device)
        prepared = prepare_combine_bias(bias_tensor(make_bias(1), device), device)
        kw = dict(num_tokens=tokens, batch=batch, memory_config=config)
        print("SHAPE", batch)
        out = nomic_moe_combine(rows, idx, w, prepared, **kw)
        ttnn.synchronize_device(device)
        t("_validate (hit)", lambda: op._validate(rows, idx, w, prepared, **kw))
        st = op._validate(rows, idx, w, prepared, **kw)
        t("check_tensors", lambda: op._check_tensors(rows, idx, w, prepared, st.padded_tokens))

        def alloc():
            o = ttnn.allocate_tensor_on_device(st.output_shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, config)

        t("allocate", alloc)
        t("bind_call", lambda: pdm.bind_call(st.descriptor, st.plan, rows, idx, w, prepared.rows, out))
        sync = lambda: ttnn.synchronize_device(device)
        t("generic_op", lambda: ttnn.generic_op([rows, idx, w, prepared.rows, out], st.descriptor), n=100, sync=sync)
        t(
            "ttnn.prim? hash",
            lambda: ttnn.compute_program_descriptor_hash(st.descriptor)
            if hasattr(ttnn, "compute_program_descriptor_hash")
            else None,
        )
        t("full call", lambda: nomic_moe_combine(rows, idx, w, prepared, **kw), n=100, sync=sync)
        ttnn.synchronize_device(device)
finally:
    ttnn.close_device(device)
