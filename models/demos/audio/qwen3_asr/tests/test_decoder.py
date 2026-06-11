"""Phase-3 PCC test: ttnn Qwen3-1.7B decoder prefill last-token logits vs golden.

Run inside the dev container with HF_MODEL set to the extracted checkpoint:
  docker exec -e TT_MESH_GRAPH_DESC_PATH=.../p150_mesh_graph_descriptor.textproto \
    -e HF_MODEL=/ttwork/qwen3_asr_text_decoder qwen3asr-dev bash -lc \
    'source /opt/venv/bin/activate && cd /work && \
     python3 models/demos/audio/qwen3_asr/tests/test_decoder.py'
"""
import os, sys
import numpy as np
import torch
import ttnn
from models.tt_transformers.tt.model_config import ModelArgs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tt"))
from qwen3_asr_decoder import Qwen3ASRDecoder  # noqa: E402

GOLDEN = os.environ.get("GOLDEN_DIR", "/golden")


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        args = ModelArgs(dev, max_batch_size=1, max_seq_len=1024)
        print("max_prefill_chunk_size =", args.max_prefill_chunk_size)
        state_dict = args.load_state_dict()
        model = Qwen3ASRDecoder(
            args, ttnn.bfloat16, dev, state_dict,
            args.weight_cache_path(ttnn.bfloat16), use_paged_kv_cache=False,
        )
        ie = torch.from_numpy(np.load(f"{GOLDEN}/inputs_embeds.npy")).float().unsqueeze(0)  # (1,S,2048)
        logits, S = model.prefill_logits(ie)
        gold = torch.from_numpy(np.load(f"{GOLDEN}/lm_head.npy")).float()[0, S - 1]  # (vocab,)
        p = pcc(logits, gold)
        print(f"[S] {S}  [logits] tt={tuple(logits.shape)} gold={tuple(gold.shape)}")
        print(f"[argmax] tt={int(logits.argmax())} gold={int(gold.argmax())}")
        print(f"[PCC] last-token logits = {p:.6f}")
        ok = (int(logits.argmax()) == int(gold.argmax())) and p > 0.97
        print("RESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    sys.exit(main())
