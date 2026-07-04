"""Phase-5 end-to-end (decoder only): prefill golden embeds -> greedy decode -> text.
Validates the full prefill+decode loop on device against the reference transcription."""
import os, sys, time
import numpy as np
import torch
import ttnn
from transformers import AutoTokenizer
from models.tt_transformers.tt.model_config import ModelArgs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tt"))
from qwen3_asr_decoder import Qwen3ASRDecoder  # noqa: E402

GOLDEN = os.environ.get("GOLDEN_DIR", "/golden")
CKPT = os.environ.get("HF_MODEL", "/ttwork/qwen3_asr_text_decoder")
REF = "What's going on? Yako-san alone for the war? Is it? War? That's when it starts. The problem is."


def main():
    tok = AutoTokenizer.from_pretrained(CKPT)
    dev = ttnn.open_device(device_id=0, trace_region_size=200000000)
    try:
        args = ModelArgs(dev, max_batch_size=1, max_seq_len=1024)
        sd = args.load_state_dict()
        model = Qwen3ASRDecoder(args, ttnn.bfloat16, dev, sd,
                                args.weight_cache_path(ttnn.bfloat16), use_paged_kv_cache=False)
        ie = torch.from_numpy(np.load(f"{GOLDEN}/inputs_embeds.npy")).float().unsqueeze(0)

        t0 = time.time()
        ids = model.generate(ie, max_new_tokens=64)
        dt = time.time() - t0

        txt = tok.decode(ids, skip_special_tokens=True).strip()
        print(f"[decode] {len(ids)} tok in {dt:.2f}s ({len(ids)/dt:.1f} tok/s)")
        print(f"[tt ] {txt!r}")
        print(f"[ref] {REF!r}")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
