"""Fast golden attention (hf.use_fast_attention) vs the cached blocked-eager golden (chain_L6_S8192: layers 0..5,
8192 tokens, fp32): per-layer time and hidden / K / V differences.

    python models/demos/mimo_v2_d_p/tests/perf/check_fast_ref.py   (THREADS=16)
"""
import os
import sys
import time

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[5]))
from models.demos.mimo_v2_d_p.reference import hf  # noqa: E402
from models.demos.mimo_v2_d_p.reference.config import MiMoTextConfig  # noqa: E402
from models.demos.mimo_v2_d_p.reference.weights import global_state, layer_state  # noqa: E402
from models.demos.mimo_v2_d_p.tests.golden import GOLDEN_DIR  # noqa: E402

torch.set_num_threads(int(os.environ.get("THREADS", "16")))
g = torch.load(GOLDEN_DIR / "chain_L6_S8192.pt")
cfg = MiMoTextConfig.from_json()
hcfg = hf.hf_config()
hf.use_fast_attention()
hf.FLASH_GA = os.environ.get("FLASH_GA", "1") == "1"
x = global_state()["embed_tokens.weight"][g["ids"]][None].float()
with torch.no_grad():
    for i in range(6):
        spec = cfg.layer_attn(i)
        t0 = time.time()
        sd = layer_state(i, cfg)
        t_load = time.time() - t0
        t0 = time.time()
        layer = hf.decoder_layer(i, sd, hcfg)
        t_build = time.time() - t0
        del sd
        t0 = time.time()
        x = hf.run_layer(layer, x, spec.window is not None, hcfg, window=spec.window, dense_mask=False)
        dt = time.time() - t0
        k, v = hf.KV_CAPTURE.pop(i)
        ref = g["hidden"][i + 1].float()
        a = x[0].bfloat16().float()
        d = (a - ref).abs().max().item()
        rel = ((a - ref).norm() / ref.norm()).item()
        kr, vr = (t.float() for t in g["kv"][i])
        print(
            f"L{i}: load {t_load:5.1f} s build {t_build:5.1f} s forward {dt:6.1f} s  hidden max|d| {d:.3e} rel {rel:.2e} | K rel {((k.bfloat16().float()-kr).norm()/kr.norm()):.2e} V rel {((v.bfloat16().float()-vr).norm()/vr.norm()):.2e}",
            flush=True,
        )
        del layer
