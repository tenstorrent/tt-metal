import ttnn, math
from eval.golden_tests.genify_matmul.feature_spec import LOOSE_CASES, FEATURE_CASES
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

dev = ttnn.open_device(device_id=0)


def alloc(shape, dtype, l1=False):
    mc = ttnn.L1_MEMORY_CONFIG if l1 else ttnn.DRAM_MEMORY_CONFIG
    return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, dev, mc)


rows = []
for c in LOOSE_CASES + FEATURE_CASES:
    if c["placement"] != "interleaved":
        continue
    ex = c["extras"]
    if ex.get("chunks") or ex.get("a_segments") or "segments" in ex:
        continue
    a_shape, b_shape = list(c["inputs"][0]), list(c["inputs"][1])
    try:
        a = alloc(a_shape, c["dtype"], ex.get("act_buffer") == "l1")
        b = alloc(b_shape, c["weight_dtype"], ex.get("weight_buffer") == "l1")
    except Exception as e:
        print("SKIP", a_shape, b_shape, e)
        continue
    ep = c["epilogue"]
    n = b_shape[-1]
    n_out = n // 2 if ep == "swiglu" else n
    odt = ex.get("output_dtype", c["dtype"])
    out = alloc(a_shape[:-1] + [n_out], odt, ex.get("out_buffer") == "l1")
    row = alloc([1, n_out], ex.get("bias_dtype", odt)) if ep in ("bias", "bias_relu", "addcmul") else None
    res = alloc(a_shape[:-1] + [n_out], odt) if ep == "addcmul" else None
    fid = ex["math_fidelity"]
    res_plans = []
    for flag in (False, True):
        pd.ACT_STREAM_BANK_LOCAL = flag
        try:
            geom, grid, blk, _ = pd.plan(
                a, b, out, c["fp32_dest_acc_en"], dev, epilogue=ep, row=row, residual=res, math_fidelity=fid
            )
            res_plans.append(
                (
                    pd.operand_mode(geom, grid, blk),
                    grid.placement,
                    grid.grid_m,
                    grid.grid_n,
                    round(pd.split_cost(geom, grid, blk) / 1350, 1),
                )
            )
        except Exception as e:
            res_plans.append(("ERR", str(e)[:60]))
    tag = "DIFF" if res_plans[0] != res_plans[1] else "same"
    print(
        tag,
        a_shape,
        b_shape,
        c["dtype"],
        c["weight_dtype"],
        ep,
        ex.get("act_buffer", ""),
        res_plans[0],
        "->",
        res_plans[1],
        "roof",
        ex.get("roofline_ns"),
    )
    for t in (a, b, out, row, res):
        if t is not None:
            t.deallocate()
ttnn.close_device(dev)
