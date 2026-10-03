"""CPU: dump the real FFN input, expert choice and routing weights of a layer's decode step (16 tokens) from the
reference chain state, for the isolated moe_compute device test.   python tests/dump_ffn_inputs.py"""
import os

import torch

from models.demos.blackhole.deepseek_v41_flash.reference import ref_kernels
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards

torch.set_num_threads(24)
CHAIN = os.environ.get("DSV41_CHAIN", "/mnt/tt-data/ssinghal/dsv4-chain-e")
ref_kernels.FAKE_QUANT = False
sh = _Shards()
for L in [int(x) for x in os.environ.get("DSV41_ANALYZE_LAYERS", "1,8").split(",")]:
    d = torch.load(os.path.join(CHAIN, f"layer_{L}.pt"))
    blk = R.build_layer(L, max_batch_size=16, max_seq_len=256)
    blk.attn.window_kv_cache.copy_(d["state"]["window"])
    cap = {}
    blk.ffn.register_forward_hook(lambda m, i, o: cap.update(x=i[0].detach(), y=o.detach()))
    blk(d["dec_in"].to(torch.bfloat16), 9, d["pre_in"], None)
    x = cap["x"].reshape(16, 5120).float()
    wg, bg = sh.get(f"layers.{L}.ffn.gate.weight").float(), sh.get(f"layers.{L}.ffn.gate.bias").float()
    s = torch.nn.functional.softplus(x @ wg.T).sqrt()
    idx = (s + bg).topk(6, -1).indices
    wt = s.gather(1, idx)
    wt = wt / wt.sum(-1, keepdim=True) * 1.5
    torch.save(
        {"x": x.to(torch.bfloat16), "idx": idx, "wt": wt, "ffn_out_ref": cap["y"].reshape(16, 5120).float()},
        os.path.join(CHAIN, f"ffn_inputs_{L}.pt"),
    )
    print(f"saved layer {L}", flush=True)
