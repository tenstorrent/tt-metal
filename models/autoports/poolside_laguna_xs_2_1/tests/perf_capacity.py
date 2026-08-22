"""Long-prefill capacity probe: run prefill beyond the single-shot SDPA limit (natural
Q-chunking) and confirm it completes with finite output. Records the largest length run."""
import json
import sys
import time

import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_reference as R
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_weights as W
from models.autoports.poolside_laguna_xs_2_1.tests.laguna_test_utils import DOC_DIR
from models.autoports.poolside_laguna_xs_2_1.tt.functional_decoder import FunctionalDecoder

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 4
SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
HIDDEN = 2048
ART = DOC_DIR / "functional_decoder"

dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    cfg = R.build_config()
    raw = W.load_layer_tensors(LAYER)
    dec = FunctionalDecoder.from_state_dict(raw, hf_config=cfg, layer_idx=LAYER, mesh_device=dev, max_seq_len=SEQ + 64)
    kv = dec.alloc_kv_cache(max_users=1, max_seq_len=SEQ + 64, block_size=32)
    pt = dec.make_page_table(1, kv["blocks_per_user"])
    torch.manual_seed(0)
    x = torch.randn(1, SEQ, HIDDEN) * 0.5
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    t0 = time.perf_counter()
    out = dec.prefill_forward(xt, kv, pt, user_id=0, start_pos=0)
    ttnn.synchronize_device(dev)
    dt = time.perf_counter() - t0
    got = ttnn.to_torch(out).float()
    finite = bool(torch.isfinite(got).all())
    res = {
        "layer": LAYER,
        "attention_type": cfg.layer_types[LAYER],
        "prefill_seq": SEQ,
        "chunk_threshold": dec.PREFILL_SDPA_CHUNK,
        "finite": finite,
        "out_shape": list(got.shape),
        "out_std": float(got.std()),
        "wall_s": round(dt, 2),
    }
    print("CAPACITY_RESULT", json.dumps(res))
    try:
        with open(ART / "prefill_capacity.json") as f:
            allres = json.load(f)
    except Exception:
        allres = []
    allres.append(res)
    with open(ART / "prefill_capacity.json", "w") as f:
        json.dump(allres, f, indent=2)
except Exception as e:
    import traceback

    traceback.print_exc()
finally:
    ttnn.close_mesh_device(dev)
