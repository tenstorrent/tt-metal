import argparse
import time

import torch

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tests import utils
from models.autoports.zai_org_glm_4_7_flash.tt.optimized_decoder import OptimizedDecoder

ap = argparse.ArgumentParser()
ap.add_argument("--attn-lat-mem", choices=["dram", "l1"], default="dram")
ap.add_argument("--batch", type=int, default=1)
args = ap.parse_args()


if args.attn_lat_mem == "l1":
    _orig = ttnn.transformer.paged_flash_multi_latent_attention_decode

    def _patched(*a, **kw):
        kw["memory_config"] = ttnn.L1_MEMORY_CONFIG
        return _orig(*a, **kw)

    ttnn.transformer.paged_flash_multi_latent_attention_decode = _patched

cfg = utils.hf_config()
layer_idx = utils.LAYER_KINDS["moe"]
sd = utils.synth_layer_state_dict(cfg, layer_idx)
B = args.batch
device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=0)
try:
    dec = OptimizedDecoder.from_state_dict(
        sd,
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=device,
        max_batch_size=B,
        max_context=4096,
        prefill_chunk_size=1024,
    )
    paged = dec.paged_config
    cache = dec.allocate_kv_cache()
    pt_torch = utils.make_page_table(B, paged.max_num_blocks, seed=3)
    pt = ttnn.from_torch(pt_torch, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    S = 1023
    x = utils.synth_activations(cfg, layer_idx, S + 2, seed=7)
    x_tt = ttnn.from_torch(x[:, :S].unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    out = dec.prefill_forward(x_tt, kv_cache=cache, page_table=pt, user_id=0, seq_len=S)
    ttnn.deallocate(out)
    ttnn.deallocate(x_tt)

    pos = S
    xd = x[:, pos : pos + 1].unsqueeze(0).permute(0, 2, 1, 3)
    if B > 1:
        xd = xd.expand(1, 1, B, xd.shape[-1]).contiguous()
    x_dev = ttnn.from_torch(xd, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    pos_dev = ttnn.from_torch(torch.tensor([pos] * B, dtype=torch.int32), device=device)
    rot_dev = ttnn.from_torch(torch.tensor([[pos]] * B, dtype=torch.uint32), device=device)

    out_c = dec.decode_forward(x_dev, kv_cache=cache, page_table=pt, cur_pos_tensor=pos_dev, rot_idxs=rot_dev)
    ttnn.deallocate(out_c)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    out_t = dec.decode_forward(x_dev, kv_cache=cache, page_table=pt, cur_pos_tensor=pos_dev, rot_idxs=rot_dev)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    for _ in range(3):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(32):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    t1 = time.perf_counter()
    ms = (t1 - t0) / 32 * 1000
    assert not torch.isnan(ttnn.to_torch(out_t)).any()
    print(f"RESULT attn_lat_mem={args.attn_lat_mem} batch={B} decode_ms_per_token={ms:.4f}")
    ttnn.release_trace(device, tid)
finally:
    ttnn.close_device(device)
