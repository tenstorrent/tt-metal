import torch, ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as pd


def run(shape, levers, dtype=ttnn.bfloat16, tile_h=None, pad=None, out_dtype=None, in_mem=None):
    saved = dict(pd.LEVERS)
    pd.LEVERS.update(levers)
    try:
        t = torch.randn(shape).to(torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32)
        x = ttnn.from_torch(
            t, dtype=dtype, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=in_mem or ttnn.DRAM_MEMORY_CONFIG
        )
        call = {}
        if tile_h:
            call["tile"] = ttnn.Tile([tile_h, 32])
        if out_dtype:
            call["dtype"] = out_dtype
        if pad:
            call.update(pad)
        out = tilize(x, **call)
        got = ttnn.to_torch(out)
        ok = torch.equal(got.float(), t.float())
        print(
            f"{levers} shape={shape} tile_h={tile_h} pad={bool(pad)} -> {'EXACT' if ok else 'MISMATCH maxdiff=' + str((got.float()-t.float()).abs().max().item())}"
        )
    finally:
        pd.LEVERS.update(saved)


for lv in [dict(read_state=1), dict(precomp_index=1), dict(read_state=1, precomp_index=1)]:
    run([1, 1, 256, 256], lv)
    run([1, 1, 64, 128], lv, dtype=ttnn.float32)
    run([1, 1, 256, 256], lv, tile_h=8)
# write_state needs a <=512B output page: tile_h=8 bf16 -> 512B. write_trid must be off.
for lv in [dict(write_state=1, write_trid=0), dict(write_state=1, write_trid=0, read_state=1, precomp_index=1)]:
    run([1, 1, 256, 256], lv, tile_h=8)
    run([1, 1, 256, 256], lv, tile_h=4)
# padded path with the new levers
run([1, 1, 50, 50], dict(read_state=1, precomp_index=1), pad=dict(output_padded_shape=[1, 1, 64, 64], pad_value=0.0))
