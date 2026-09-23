import torch, ttnn

device = ttnn.open_device(device_id=0)
cases = [
    ([1, 1, 50, 50], [1, 1, 64, 64]),
    ([1, 1, 50, 64], [1, 1, 64, 64]),
    ([1, 1, 32, 64], [1, 1, 64, 128]),
    ([1, 1, 32, 64], [2, 1, 32, 64]),
    ([64], [32, 64]),
    ([], [32, 32]),
    ([1, 1, 30, 32], [1, 1, 32, 32]),
    ([3, 50, 64], [3, 64, 64]),
]
for logical, padded in cases:
    try:
        x = torch.randn(padded).bfloat16()
        t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        a = t.buffer_address()
        v = ttnn.reshape(t, ttnn.Shape(logical), ttnn.Shape(padded))
        print(
            "CASE",
            logical,
            padded,
            "shape",
            list(v.shape),
            "padded",
            list(v.padded_shape),
            "addr_same",
            v.buffer_address() == a,
        )
        lg = ttnn.to_torch(v)
        pd = v.cpu().to_torch_with_padded_shape()
        exp_l = x
        for d in range(len(padded)):
            pass
        sl = tuple(slice(0, n) for n in ([1] * (len(padded) - len(logical)) + list(logical)))
        print(
            "CASE eq_logical",
            torch.equal(lg.reshape(x[sl].shape) if lg.numel() == x[sl].numel() else lg, x[sl]),
            "lg",
            list(lg.shape),
            "eq_padded",
            torch.equal(pd, x),
            list(pd.shape),
        )
    except Exception as e:
        print("CASE ERR", logical, padded, str(e)[:300])
ttnn.close_device(device)
