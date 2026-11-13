"""attn-bfp4: oracle attention taps at deeper layers (CPU, fp32 oracle, S tokens). Advances the iHC stream through every layer; saves taps for the chosen layers.
  PYTHONPATH=<wt>:... python abf4_gen_deep.py OUT_DIR TOKENS L1,L2,...
"""
import os
import sys
import time

import torch

import models.demos.blackhole.hy4_preview as _p

_p.__path__.append("/mnt/tt-data/ssinghal/wt/hy4_ref/models/demos/blackhole/hy4_preview")
from models.demos.blackhole.hy4_preview.reference import oracle as O
from models.demos.blackhole.hy4_preview.reference.dump_ref import long_prompt
from models.demos.blackhole.hy4_preview.reference.oracle import Conv, Hy4Oracle
from models.demos.blackhole.hy4_preview.tests.gen_attn_golden import attn_taps

out_dir, n_tok, keep = sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")]
torch.set_num_threads(int(os.environ.get("HY4_THREADS", "24")))
os.makedirs(out_dir, exist_ok=True)
from tokenizers import Tokenizer

tok = Tokenizer.from_file("/mnt/tt-data/Hy4-Preview/tokenizer.json")
ids = [120000] + tok.encode(long_prompt(120)).ids
ids = torch.tensor([ids[:n_tok]])
conv = Conv()
o = Hy4Oracle()
x4 = o.embed_ids(ids)
st = {"idx_source": -1}
t0 = time.time()
for L in range(max(keep) + 1):
    if L in keep:
        tp = attn_taps(o.ck, L, x4, st, conv)
        torch.save(
            {k: tp[k] for k in ("h", "qa", "o", "gate", "out", "idx", "idx_source")}, os.path.join(out_dir, f"L{L}.pt")
        )
        print(f"layer {L} taps {time.time()-t0:.0f}s", flush=True)
    x4 = O.decoder_layers_multi(o.ck, L, [x4], [conv], [st])[0]
    print(f"layer {L} advanced {time.time()-t0:.0f}s norm {x4.norm(dim=-1).mean().item():.3f}", flush=True)
