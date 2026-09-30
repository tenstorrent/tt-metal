import torch, ttnn
from ttnn.operations.mhc_post import mhc_post

dev = ttnn.open_device(device_id=0)
T, C, N = 32, 32, 4
torch.set_printoptions(linewidth=200, precision=3)


def run(f, x, post, comb):
    d = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    out = ttnn.to_torch(mhc_post(d(f), d(x), d(post), d(comb))).float().reshape(T, N, C)
    ref = post.reshape(T, N, 1) * f.reshape(T, 1, C) + torch.einsum(
        "tij,tic->tjc", comb.reshape(T, N, N), x.reshape(T, N, C)
    )
    return out, ref


# test 1: F=1, X=0, post[t,j] = 10*j + t/32 -> out_j = post_j
f = torch.ones(T, C)
x = torch.zeros(T, N * C)
post = torch.tensor([[10.0 * j + t / 32 for j in range(N)] for t in range(T)])
comb = torch.zeros(T, N * N)
out, ref = run(f, x, post, comb)
for j in range(N):
    print("post test j", j, "maxdiff", (out[:, j] - ref[:, j]).abs().max().item())
    print(
        " rows 0,15,16,31 cols 0,16:",
        out[[0, 15, 16, 31]][:, j][:, [0, 16]].flatten().tolist(),
        "ref",
        ref[[0, 15, 16, 31]][:, j][:, [0, 16]].flatten().tolist(),
    )
# test 2: F=0, X_i = 1, comb[i][j] = 100*i + j -> out_j = sum_i comb
f = torch.zeros(T, C)
x = torch.ones(T, N * C)
post = torch.zeros(T, N)
comb = torch.tensor([[100.0 * (m // N) + (m % N) for m in range(N * N)] for t in range(T)])
out, ref = run(f, x, post, comb)
for j in range(N):
    print(
        "comb test j",
        j,
        "maxdiff",
        (out[:, j] - ref[:, j]).abs().max().item(),
        "got",
        out[0, j, 0].item(),
        out[0, j, 16].item(),
        out[20, j, 0].item(),
        "ref",
        ref[0, j, 0].item(),
    )
# test 3: single X_i hot
for i in range(N):
    x = torch.zeros(T, N * C)
    x[:, i * C : (i + 1) * C] = 1
    out, ref = run(f, x, post, comb)
    print(
        "X",
        i,
        "hot: got j0..3",
        [out[0, j, 0].item() for j in range(N)],
        "ref",
        [ref[0, j, 0].item() for j in range(N)],
    )
ttnn.close_device(dev)
