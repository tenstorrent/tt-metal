import torch, ttnn
from ttnn.operations.tilize import tilize

device = ttnn.open_device(device_id=0)
try:

    def run(shape, pad_value=0.0, target=None, label=""):
        x = (
            torch.arange(int(torch.tensor(shape).prod()) if shape else 1, dtype=torch.float32)
            .reshape(shape or [])
            .to(torch.bfloat16)
        )
        ti = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        kw = {"pad_value": pad_value}
        if target is not None:
            kw["output_padded_shape"] = target
        out = tilize(ti, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16, **kw)
        # logical
        lg = ttnn.to_torch(out)
        ok_logical = torch.equal(lg, x)
        pad = out.cpu().to_torch_with_padded_shape()
        # expected
        xs = (
            x.reshape((1,) * (len(list(pad.shape)) - x.dim()) + tuple(x.shape)) if x.dim() < len(list(pad.shape)) else x
        )
        pads = tuple(j for i in reversed(range(xs.dim())) for j in (0, list(pad.shape)[i] - xs.shape[i]))
        exp = torch.nn.functional.pad(xs, pads, value=pad_value)
        ok_pad = torch.equal(pad, exp)
        print(
            f"{label or shape}: padded_shape={list(pad.shape)} logical_ok={ok_logical} pad_ok={ok_pad}"
            + ("" if ok_pad else f"  ndiff={(pad!=exp).sum().item()}/{exp.numel()}")
        )

    run([1, 1, 32, 50], 0.0, None, "w_tail")
    run([1, 1, 50, 64], 3.0, None, "h_tail")
    run([1, 1, 50, 50], -7.0, None, "hw_tail")
    run([1, 1, 30, 32], 0.0, None, "subtile_both")
    run([50, 50], 0.0, None, "rank2")
    run([3, 50, 64], 2.0, None, "rank3")
    run([64], 0.0, None, "rank1")
    run([], 0.0, None, "rank0")
    run([1, 1, 32, 50], 0.0, [1, 1, 32, 128], "explicit_w")
    run([1, 1, 50, 50], 5.0, [1, 1, 128, 128], "explicit_hw")
    run([1, 1, 32, 64], 0.0, [1, 1, 64, 128], "explicit_whole_tiles")
    run([1, 1, 30, 32], -4.0, [1, 1, 32, 32], "explicit_exact")
    run([1, 1, 1, 2048], 0.0, None, "single_stick")
    run([1, 1, 32, 4090], 0.0, None, "short_wide_w_tail")
finally:
    ttnn.close_device(device)
