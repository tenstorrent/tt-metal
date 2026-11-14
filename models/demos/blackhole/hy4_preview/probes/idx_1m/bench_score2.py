# Decode score alternatives at n_local keys: stock op vs two-matmul chain (heads as M rows: no 32x row padding), key dtypes. Prints R lines. Env NS
import json

import torch
from idx_common import *

import ttnn


def main():
    ns = [int(x) for x in os.environ.get("NS", "32768,131072").split(",")]
    md = open_md()
    gen = torch.Generator().manual_seed(2)
    for n in ns:
        kt = torch.randn(8, 1, n, DIM, generator=gen).to(torch.bfloat16)
        for kn, kdt in (("bf8", ttnn.bfloat8_b), ("bf4", ttnn.bfloat4_b)):
            k = up(md, kt, kdt, shard=True)
            for S in (1, 2):
                q = torch.randn(1, 1, S * 32, DIM, generator=gen).to(
                    torch.bfloat16
                )  # S query rows x 32 heads as M rows
                w = torch.zeros(1, 1, 32, S * 32)
                for s in range(S):
                    w[0, 0, s, s * 32 : (s + 1) * 32] = torch.rand(32, generator=gen) * 0.2
                qd, wd = up(md, q), up(md, w.to(torch.bfloat16))
                for fid in ("HiFi4", "HiFi2", "LoFi"):
                    ckc = ttnn.init_device_compute_kernel_config(
                        md.arch(),
                        math_fidelity=getattr(ttnn.MathFidelity, fid),
                        math_approx_mode=False,
                        fp32_dest_acc_en=(fid == "HiFi4"),
                        packer_l1_acc=False,
                    )
                    r = {"n": n, "k": kn, "S": S, "fid": fid}
                    try:

                        def f():
                            s1 = ttnn.matmul(
                                qd, k, transpose_b=True, activation="relu", compute_kernel_config=ckc
                            )  # [1,1,S*32,n]
                            s2 = ttnn.matmul(wd, s1, compute_kernel_config=ckc)  # [1,1,32,n]
                            return ttnn.to_layout(ttnn.slice(s2, [0, 0, 0, 0], [1, 1, S, n]), ttnn.ROW_MAJOR_LAYOUT)

                        r["chain_us"] = round(traced_us(md, f, n=3, reps=2), 1)
                        r["mm1_us"] = round(
                            traced_us(
                                md,
                                lambda: ttnn.matmul(
                                    qd, k, transpose_b=True, activation="relu", compute_kernel_config=ckc
                                ),
                                n=3,
                                reps=2,
                            ),
                            1,
                        )
                    except Exception as e:
                        r["err"] = str(e).split("\n")[0][:160]
                    print("R", json.dumps(r), flush=True)
            ttnn.deallocate(k)
    close_md(md)


main()
