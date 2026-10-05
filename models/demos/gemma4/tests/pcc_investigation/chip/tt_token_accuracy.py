"""Standard tt-metal token-accuracy run (simple_text_demo ci-token-matching / TokenAccuracy) for Gemma 4.

Same protocol as models/tt_transformers/demo/simple_text_demo.py::TokenAccuracy:
  - reference: a .refpt from generate_reference_hf.py (Tale of Two Cities, 1024 tokens, HF top-5 per position)
  - prompt = reference_tokens[:512]; then decode with teacher forcing (each step is fed the real next token)
  - top-1 = model argmax == HF top-1; top-5 = model argmax in HF top-5
Model setup is the accuracy gate's (Gemma4Generator, paged KV, untraced, logits back to host).
Also reports accuracy against the real text and per-position PCC against the saved HF logits.
"""

import os as _os
from pathlib import Path as _Path

# Where things live (override with env vars): investigation data, model checkpoints, repo root.
DATA = _os.environ.get("GEMMA4_PCC_DATA", _os.path.expanduser("~/benchmark-data/gemma4-pcc"))
MODELS = _os.environ.get("GEMMA4_PCC_MODELS", _os.path.expanduser("~/benchmark-data"))
REPO = _os.environ.get("GEMMA4_PCC_REPO") or str(_Path(__file__).resolve().parents[6])

import gc
import json
import math
import sys
from pathlib import Path

import torch

import ttnn
from models.tt_transformers.tests.optimizer_weight_cache import RunCache

MODEL = f"{MODELS}/gemma-4-26B-A4B-it"
HERE = Path(DATA)
REFPT = HERE / "gemma-4-26B-A4B-it.refpt"
MAX_SEQ_LEN = 1024
PAGE_BLOCK_SIZE = 32
GENERATED_TOKENS = 500  # the ci-token-matching case's max_generated_tokens


def pcc(a, b):
    return float(torch.corrcoef(torch.stack((a.float().reshape(-1), b.float().reshape(-1))))[0, 1])


def main(label):
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.generator_trace import resolve_gemma4_demo_long_context
    from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
    from models.tt_transformers.tt.common import PagedAttentionConfig

    ref = torch.load(REFPT)
    reference_tokens = ref["reference_tokens"][0]  # [1024]
    split = reference_tokens.shape[-1] // 2  # 512, as TokenAccuracy
    prompt = reference_tokens[:split]
    continuation = reference_tokens[split:]
    top5 = ref["top5_tokens"][split - 1 :, :]  # HF top-5 for the token after each position, from position 511 on
    hf_logits = torch.load(str(REFPT) + ".logits.pt")  # [1023, vocab] bf16

    cache = RunCache(Path(f"{REPO}/generated/optimizer_cache"), f"gemma4-tokacc-{label}-")
    import os

    os.environ["TT_CACHE_PATH"] = cache.path
    mesh_device = generator = tt_kv_cache = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=24576, num_command_queues=1)
        pac = PagedAttentionConfig(block_size=PAGE_BLOCK_SIZE, max_num_blocks=math.ceil(MAX_SEQ_LEN / PAGE_BLOCK_SIZE))
        lc = resolve_gemma4_demo_long_context(MAX_SEQ_LEN, mesh_device, MODEL, paged_attention=True)
        generator, tt_kv_cache, _ = cache.build(
            lambda: Gemma4Generator.from_pretrained(
                mesh_device=mesh_device,
                model_path=MODEL,
                max_batch_size=1,
                max_seq_len=MAX_SEQ_LEN,
                paged_attention_config=pac,
                bounded_sliding_kv_cache=lc["bounded_sliding"],
            ),
            loaders=[(Gemma4ModelArgs, "load_state_dict")],
        )
        gc.collect()
        cache.loaded()
        vocab = generator.model_args[0].vocab_size
        page_table = torch.arange(pac.max_num_blocks, dtype=torch.int32).reshape(1, pac.max_num_blocks)

        def host(out):
            first = out[0] if isinstance(out, (tuple, list)) else out
            return first.float().reshape(-1, first.shape[-1])[0, :vocab]

        rows = []  # (position, logits)
        logits = host(
            generator.prefill_forward_text(
                prompt.reshape(1, -1).long(),
                page_table=page_table,
                kv_cache=tt_kv_cache,
                prompt_lens=[split],
                warmup_prefill=False,
                enable_trace=False,
                sampling_params=None,
            )
        )
        rows.append((split - 1, logits))
        for step in range(GENERATED_TOKENS - 1):
            pos = split + step  # feed the real token at this position (teacher forcing)
            logits = host(
                generator.decode_forward(
                    continuation[step].reshape(1, 1).long(),
                    torch.tensor([pos], dtype=torch.int64),
                    page_table=page_table,
                    kv_cache=tt_kv_cache,
                    enable_trace=False,
                    sampling_params=None,
                )
            )
            rows.append((pos, logits))

        n = len(rows)
        pred = torch.tensor([int(l.argmax()) for _, l in rows])
        t1 = (pred == top5[:n, 0]).float().mean().item() * 100
        t5 = (pred[:, None] == top5[:n]).any(1).float().mean().item() * 100
        real = continuation[:n]
        real_acc = (pred == real).float().mean().item() * 100
        hf_real_acc = (top5[:n, 0] == real).float().mean().item() * 100
        pccs = [pcc(l, hf_logits[p]) for p, l in rows]
        result = {
            "label": label,
            "positions": n,
            "standard_top1_vs_hf_pct": round(t1, 2),
            "standard_top5_vs_hf_pct": round(t5, 2),
            "chip_top1_vs_real_text_pct": round(real_acc, 2),
            "hf_top1_vs_real_text_pct": round(hf_real_acc, 2),
            "mean_pcc_vs_hf": round(sum(pccs) / n, 6),
            "min_pcc_vs_hf": round(min(pccs), 6),
            "positions_pcc_below_0_99": sum(p < 0.99 for p in pccs),
        }
        print("TOKACC " + json.dumps(result), flush=True)
        torch.save({"pred": pred, "pccs": pccs}, HERE / f"tt-{label}.pt")
        torch.save({"positions": [p for p, _ in rows], "logits": torch.stack([l for _, l in rows]).to(torch.bfloat16)}, HERE / f"tt-{label}-logits.pt")
    finally:
        generator = tt_kv_cache = None
        gc.collect()
        if mesh_device is not None:
            ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        cache.cleanup()


if __name__ == "__main__":
    main(sys.argv[1])
