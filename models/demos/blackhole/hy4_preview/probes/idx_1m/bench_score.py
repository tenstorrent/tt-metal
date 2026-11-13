# indexer_score_dsa tuning at n_local=131072 (1M/8): program config, dtypes, fidelity. Sq=32 (decode) and Sq=512 (prefill)
import itertools
import json

import torch
from idx_common import *

import ttnn


def main():
    n = int(os.environ.get("N", "131072"))
    md = open_md()
    kd = {"bf8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16, "bf4": ttnn.bfloat4_b}
    kt = torch.randn(8, 1, n, DIM).to(torch.bfloat16)
    ks = {name: up(md, kt, dt, shard=True) for name, dt in kd.items() if name != "bf16" or n <= 65536}
    for Sq in (32, 512):
        for qdn in ("bf16", "bf8"):
            q = up(md, torch.randn(1, HEADS, Sq, DIM).to(torch.bfloat16), kd[qdn])
            w = up(md, torch.rand(1, 1, Sq, HEADS).to(torch.bfloat16))
            for kn, k in ks.items():
                for qc, kc, hg in itertools.product((32, 64) if Sq == 512 else (32,), (128, 256, 512), (0, 16)):
                    for fid in ("def", "LoFi"):
                        r = {"Sq": Sq, "n": n, "q": qdn, "k": kn, "qc": qc, "kc": kc, "hg": hg, "fid": fid}
                        try:
                            cfg = ttnn.IndexerScoreProgramConfig(q_chunk_size=qc, k_chunk_size=kc, head_group_size=hg)
                            ckc = (
                                None
                                if fid == "def"
                                else ttnn.init_device_compute_kernel_config(
                                    md.arch(), math_fidelity=ttnn.MathFidelity.LoFi
                                )
                            )
                            f = lambda: ttnn.experimental.indexer_score_dsa(
                                q,
                                k,
                                w,
                                chunk_start_idx=n - 32,
                                kv_len=n,
                                program_config=cfg,
                                compute_kernel_config=ckc,
                                cache_batch_idx=0,
                            )
                            r["us"] = round(traced_us(md, f, n=3, reps=2), 1)
                        except Exception as e:
                            r["err"] = str(e).split("\n")[0][:120]
                        print("R", json.dumps(r), flush=True)
    close_md(md)


main()
