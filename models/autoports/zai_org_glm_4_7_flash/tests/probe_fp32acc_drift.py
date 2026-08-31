import sys
import time

import ttnn

sys.path.insert(0, "/home/stisi/tt-metal")
from models.autoports.zai_org_glm_4_7_flash.tests import utils
from models.autoports.zai_org_glm_4_7_flash.tt.functional_decoder import FunctionalDecoder, PagedCacheConfig, _ck

ARM = sys.argv[1] if len(sys.argv) > 1 else "fp32acc"
S_FULL = 202752
S = S_FULL - 1
cfg = utils.hf_config()
sd = utils.synth_layer_state_dict(cfg, 0)  # dense control
layer = utils.build_hf_layer(cfg, 0, sd)
x = utils.synth_activations(cfg, 0, S_FULL, seed=7)

device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=0)
try:
    paged = PagedCacheConfig.for_context(S_FULL, 1)
    dec = FunctionalDecoder.from_state_dict(
        sd,
        hf_config=cfg,
        layer_idx=0,
        mesh_device=device,
        max_batch_size=1,
        max_context=S_FULL,
        paged_config=paged,
        prefill_chunk_size=2048,
    )
    # Arms against the FINAL implementation (ck_flash_prefill is the knob the
    # prefill flash call site reads): "baseline" reproduces the original bf16
    # accumulator drift; "fp32acc" is the shipped configuration.
    if ARM == "baseline":
        dec.ck_flash_prefill = _ck(device, ttnn.MathFidelity.HiFi4, False)
    elif ARM == "fp32acc":
        dec.ck_flash_prefill = _ck(device, ttnn.MathFidelity.HiFi4, True)
    print("ARM:", ARM)
    cache = dec.allocate_kv_cache()
    pt_torch = utils.make_page_table(1, paged.max_num_blocks, seed=3)
    pt = ttnn.from_torch(pt_torch, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    x_tt = ttnn.from_torch(x[:, :S].unsqueeze(0), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    t0 = time.perf_counter()
    out = dec.prefill_forward(x_tt, kv_cache=cache, page_table=pt, user_id=0, seq_len=S)
    ttnn.synchronize_device(device)
    print(f"prefill wall {time.perf_counter()-t0:.0f}s")
    got = ttnn.to_torch(out).float()[0, 0]
    kvpe_ref = utils.torch_latent_cache_reference(cfg, sd, x[0])
    for name, r0 in (("middle", 101376), ("end", S - 32)):
        rows = list(range(r0, r0 + 32))
        ref_rows = utils.torch_absorbed_window_reference(cfg, sd, layer, x[0], kvpe_ref, rows)
        print(f"{name}: agg PCC {utils.pcc(ref_rows, got[rows]):.6f}")
finally:
    ttnn.close_device(device)
