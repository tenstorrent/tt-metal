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


def t(name, fn, n=200):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    print(f"PIECE {name}: {1e6*ts[n//2]:.2f} us")


device = ttnn.open_device(device_id=0, l1_small_size=0)
try:
    batch, seqlen = 8, 512
    tokens = batch * seqlen
    shape = (batch, 1, seqlen, 768)
    config = output_config("norm_layout", shape, device)
    plan = route(tokens, "real_text", seed=1)
    rows = rows_tensor(make_expert_rows(plan["rows"], plan["slot_row"], seed=1), device)
    idx = index_tensor(plan["combine_index"], device)
    w = weights_tensor(plan["combine_weights"], device)
    prepared = prepare_combine_bias(bias_tensor(make_bias(1), device), device)
    kw = dict(num_tokens=tokens, batch=batch, memory_config=config)
    t("validate", lambda: op.validate(rows, idx, w, prepared, **kw))
    t("output_layout", lambda: op._output_layout(config, shape, device))
    t("repr(config)", lambda: repr(config))
    t("hash(config)", lambda: hash(config))
    t("config==config", lambda: config == config)
    t("rows.shape tuple", lambda: tuple(rows.shape))
    t("rows.memory_config()", lambda: rows.memory_config())
    t("rows.dtype", lambda: rows.dtype)
    t("rows.layout", lambda: rows.layout)
    t("rows.device()", lambda: rows.device())
    t("buffer_address", lambda: rows.buffer_address())
    t("grid", lambda: device.compute_with_storage_grid_size())

    def alloc():
        o = ttnn.allocate_tensor_on_device(
            ttnn.Shape([batch, 1, seqlen, 768]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, config
        )
        ttnn.deallocate(o)

    t("allocate+dealloc", alloc)
    o = ttnn.allocate_tensor_on_device(
        ttnn.Shape([batch, 1, seqlen, 768]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, config
    )
    t(
        "cb_from_sharded",
        lambda: ttnn.cb_descriptor_from_sharded_tensor(16, o, core_ranges=op._plan(tokens, batch, config).cores),
    )
    from ttnn.operations.nomic_moe_combine.nomic_moe_combine_program_descriptor import create_program_descriptor

    p = op._plan(tokens, batch, config)
    d = create_program_descriptor(p, rows, idx, w, prepared.rows, o, op.default_compute_kernel_config())
    t(
        "build descriptor",
        lambda: create_program_descriptor(p, rows, idx, w, prepared.rows, o, op.default_compute_kernel_config()),
        n=50,
    )

    def go():
        ttnn.generic_op([rows, idx, w, prepared.rows, o], d)

    t("generic_op (prebuilt)", go, n=100)
    ttnn.synchronize_device(device)

    def full():
        out = nomic_moe_combine(rows, idx, w, prepared, **kw)

    t("full call", full, n=50)
    ttnn.synchronize_device(device)
finally:
    ttnn.close_device(device)
